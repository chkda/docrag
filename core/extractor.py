import fitz


class Extractor:
    def __init__(self, document_path: str = None, mode: str = "page"):
        self.document_path = document_path
        self.mode = mode
        self.doc = fitz.open(document_path) if document_path else None

    def set_doc(self, document_path: str, mode: str = "page"):
        self.document_path = document_path
        self.mode = mode
        self.doc = fitz.open(document_path)

    def extract(self):
        if self.mode == "page":
            yield from self._extract_by_page()
        elif self.mode == "section":
            yield from self._extract_by_section()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'page' or 'section'")

    def _extract_by_page(self):
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            yield {
                "page_number": page_num + 1,
                "text": text
            }

    def _extract_by_section(self):
        toc = self.doc.get_toc()

        if not toc:
            yield from self._extract_by_page()
            return

        for i, (level, title, page_num) in enumerate(toc):
            start_page = page_num - 1

            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                end_page = len(self.doc)

            section_text = ""
            for page_idx in range(start_page, end_page):
                if page_idx < len(self.doc):
                    page = self.doc[page_idx]
                    section_text = page.get_text()

                    yield {
                        "section_title": title,
                        "page_number": page_idx,
                        "text": section_text
                    }

    def close(self):
        self.doc.close()
