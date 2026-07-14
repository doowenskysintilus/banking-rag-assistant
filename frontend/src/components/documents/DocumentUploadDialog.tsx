import { useRef, useState } from "react";
import { toast } from "sonner";
import { Loader2, Upload } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useUploadDocument } from "@/hooks/useDocuments";
import { ApiError } from "@/lib/api-client";

const ACCEPTED_EXTENSIONS = ".pdf,.docx,.xlsx,.xls,.txt,.html,.htm";

export function DocumentUploadDialog() {
  const [open, setOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const upload = useUploadDocument();

  async function handleUpload() {
    if (!selectedFile) return;
    try {
      await upload.mutateAsync(selectedFile);
      toast.success(`${selectedFile.name} was submitted for ingestion.`);
      setOpen(false);
      setSelectedFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : "Upload failed.");
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <Upload />
          Upload document
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Upload a document</DialogTitle>
          <DialogDescription>
            PDF, DOCX, Excel, TXT, or HTML. It will be extracted, chunked, and indexed in the background —
            watch its status on the Documents page.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2">
          <Label htmlFor="document-file">File</Label>
          <input
            id="document-file"
            ref={inputRef}
            type="file"
            accept={ACCEPTED_EXTENSIONS}
            onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
            className="block w-full text-sm text-muted-foreground file:mr-3 file:rounded-md file:border-0 file:bg-secondary file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-secondary-foreground hover:file:bg-secondary/80"
          />
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleUpload} disabled={!selectedFile || upload.isPending}>
            {upload.isPending && <Loader2 className="animate-spin" />}
            Upload
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
