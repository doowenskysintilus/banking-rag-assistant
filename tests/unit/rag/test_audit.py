import json

from sqlalchemy import select

from backend.models.audit_log import AuditLog
from backend.models.query_log import RagQueryLog
from backend.rag.audit import RagQueryAuditor
from tests.unit.rag.conftest import make_user


async def _get_rag_query_log(db_session, user_id: str) -> RagQueryLog:
    result = await db_session.execute(
        select(RagQueryLog).where(RagQueryLog.user_id == user_id)
    )
    return result.scalars().one()


async def _get_audit_log(db_session, user_id: str) -> AuditLog:
    result = await db_session.execute(
        select(AuditLog).where(AuditLog.user_id == user_id)
    )
    return result.scalars().one()


class TestSuccessfulQueryLogging:
    async def test_persists_a_rag_query_log_row(self, db_session):
        owner = await make_user(db_session)
        auditor = RagQueryAuditor()

        await auditor.log(
            db_session,
            user_id=owner.id,
            conversation_id=None,
            question="What is the overdraft fee?",
            retrieval_engine="native",
            num_context_chunks=2,
            confidence_score=0.85,
            confidence_level="high",
            citations=[{"chunk_id": "c1"}],
            documents_used=[
                {"document_id": "d1", "filename": "policy.pdf", "chunk_count": 1}
            ],
            model="gpt-4o-test",
            input_tokens=100,
            output_tokens=20,
            stage_durations_ms={"retrieval_ms": 12.5, "llm_ms": 200.0},
            total_ms=250.0,
        )

        entry = await _get_rag_query_log(db_session, owner.id)
        assert entry.question == "What is the overdraft fee?"
        assert entry.retrieval_engine == "native"
        assert entry.num_context_chunks == 2
        assert entry.confidence_score == 0.85
        assert entry.confidence_level == "high"
        assert json.loads(entry.citations_json) == [{"chunk_id": "c1"}]
        assert json.loads(entry.documents_used_json)[0]["filename"] == "policy.pdf"
        assert entry.model == "gpt-4o-test"
        assert entry.input_tokens == 100
        assert entry.output_tokens == 20
        assert json.loads(entry.stage_durations_json)["llm_ms"] == 200.0
        assert entry.total_ms == 250.0
        assert entry.error is None

    async def test_also_writes_a_generic_compliance_audit_log_entry(self, db_session):
        owner = await make_user(db_session)
        auditor = RagQueryAuditor()

        await auditor.log(
            db_session,
            user_id=owner.id,
            conversation_id="conv-123",
            question="A question",
            retrieval_engine="native",
        )

        entry = await _get_audit_log(db_session, owner.id)
        assert entry.action == "rag_query"
        assert entry.resource == "conv-123"

    async def test_redacts_pii_in_the_stored_question(self, db_session):
        owner = await make_user(db_session)
        auditor = RagQueryAuditor()

        await auditor.log(
            db_session,
            user_id=owner.id,
            conversation_id=None,
            question="My email is jane.doe@example.com, what is my balance?",
            retrieval_engine="native",
        )

        entry = await _get_rag_query_log(db_session, owner.id)
        assert "jane.doe@example.com" not in entry.question
        assert "REDACTED_EMAIL" in entry.question


class TestFailedQueryLogging:
    async def test_persists_the_error_and_marks_the_audit_action_as_failed(
        self, db_session
    ):
        owner = await make_user(db_session)
        auditor = RagQueryAuditor()

        await auditor.log(
            db_session,
            user_id=owner.id,
            conversation_id=None,
            question="A question that failed",
            retrieval_engine="native",
            error="OpenAI API timed out",
        )

        rag_entry = await _get_rag_query_log(db_session, owner.id)
        assert rag_entry.error == "OpenAI API timed out"

        audit_entry = await _get_audit_log(db_session, owner.id)
        assert audit_entry.action == "rag_query_failed"


class TestGracefulDegradation:
    async def test_logging_failure_does_not_raise(self, db_session, monkeypatch):
        owner = await make_user(db_session)
        auditor = RagQueryAuditor()

        async def _broken_commit():
            raise RuntimeError("db is down")

        monkeypatch.setattr(db_session, "commit", _broken_commit)

        # Must not raise, even though the underlying commit fails.
        await auditor.log(
            db_session,
            user_id=owner.id,
            conversation_id=None,
            question="q",
            retrieval_engine="native",
        )
