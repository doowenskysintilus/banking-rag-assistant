from backend.core.audit import append_audit_log, verify_audit_chain
from backend.models.audit_log import AuditLog


class TestAppendAuditLog:
    async def test_first_entry_has_no_prev_hash(self, db_session):
        await append_audit_log(db_session, user_id="u1", action="login_success")
        await db_session.commit()

        from sqlalchemy import select

        entry = (await db_session.execute(select(AuditLog))).scalars().one()
        assert entry.prev_hash is None
        assert entry.entry_hash

    async def test_second_entry_chains_to_the_first(self, db_session):
        first = await append_audit_log(db_session, user_id="u1", action="login_success")
        await db_session.commit()
        second = await append_audit_log(db_session, user_id="u1", action="logout")
        await db_session.commit()

        assert second.prev_hash == first.entry_hash

    async def test_detail_is_stored_encrypted_and_readable_back(self, db_session):
        from sqlalchemy import select, text

        await append_audit_log(
            db_session,
            user_id="u1",
            action="user_registered",
            detail="plain detail text",
        )
        await db_session.commit()

        # Reading back through the ORM transparently decrypts (EncryptedString).
        entry = (await db_session.execute(select(AuditLog))).scalars().one()
        assert entry.detail == "plain detail text"

        # Raw SQL bypasses the ORM's type decorator entirely — this is what's
        # actually sitting in the column, and it must not be the plaintext.
        raw_value = (
            await db_session.execute(
                text("SELECT detail FROM audit_logs WHERE id = :id"), {"id": entry.id}
            )
        ).scalar_one()
        assert raw_value != "plain detail text"
        assert raw_value.startswith("v1:")


class TestVerifyAuditChain:
    async def test_empty_table_has_no_problems(self, db_session):
        assert await verify_audit_chain(db_session) == []

    async def test_intact_chain_reports_no_problems(self, db_session):
        await append_audit_log(db_session, user_id="u1", action="login_success")
        await db_session.commit()
        await append_audit_log(db_session, user_id="u1", action="logout")
        await db_session.commit()
        await append_audit_log(db_session, user_id="u2", action="login_success")
        await db_session.commit()

        assert await verify_audit_chain(db_session) == []

    async def test_tampering_with_a_rows_action_breaks_the_chain(self, db_session):
        from sqlalchemy import select

        await append_audit_log(db_session, user_id="u1", action="login_success")
        await db_session.commit()
        await append_audit_log(db_session, user_id="u1", action="logout")
        await db_session.commit()

        first_entry = (
            (
                await db_session.execute(
                    select(AuditLog).order_by(AuditLog.created_at.asc())
                )
            )
            .scalars()
            .first()
        )
        first_entry.action = "login_success_TAMPERED"
        await db_session.commit()

        problems = await verify_audit_chain(db_session)
        assert problems  # the recomputed hash of the tampered row no longer matches
        assert any("entry_hash does not match" in p for p in problems)

    async def test_deleting_a_middle_row_breaks_the_chain_link(self, db_session):
        from sqlalchemy import select, delete

        await append_audit_log(db_session, user_id="u1", action="event_a")
        await db_session.commit()
        await append_audit_log(db_session, user_id="u1", action="event_b")
        await db_session.commit()
        await append_audit_log(db_session, user_id="u1", action="event_c")
        await db_session.commit()

        rows = (
            (
                await db_session.execute(
                    select(AuditLog).order_by(AuditLog.created_at.asc())
                )
            )
            .scalars()
            .all()
        )
        middle_id = rows[1].id
        await db_session.execute(delete(AuditLog).where(AuditLog.id == middle_id))
        await db_session.commit()

        problems = await verify_audit_chain(db_session)
        assert any("prev_hash does not match" in p for p in problems)
