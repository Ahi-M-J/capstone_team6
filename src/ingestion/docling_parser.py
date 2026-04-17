import os
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# ✅ VLM
from google import genai


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# ✅ Image folder
IMAGE_DIR = "data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)


# 🔥 VLM function
def generate_image_description(pil_img):
    try:
        model = genai.GenerativeModel("gemini-3.1-pro-preview")

        response = model.generate_content([
            "Describe this image in detail for semantic search.",
            pil_img
        ])

        return response.text.strip() if response.text else "No description available"

    except Exception as e:
        return f"Image description failed: {str(e)}"


def parse_document(file_path: str) -> list[dict]:

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    result = converter.convert(file_path)
    doc = result.document

    parsed_chunks: list[dict] = []
    current_section: str | None = None
    source_file = os.path.basename(file_path)

    for item in doc.iterate_items():
        node = item[0] if isinstance(item, tuple) else item
        label = str(getattr(node, "label", "")).lower()

        if label in ("page_header", "page_footer"):
            continue

        prov = getattr(node, "prov", None)
        page_no = prov[0].page_no if prov else None

        position = None
        if prov and hasattr(prov[0], "bbox") and prov[0].bbox:
            b = prov[0].bbox
            position = {"l": b.l, "t": b.t, "r": b.r, "b": b.b}

        # ✅ metadata WITHOUT image_path
        def _make_metadata(content_type: str, element_type: str):
            return {
                "content_type": content_type,
                "element_type": element_type,
                "section": current_section,
                "page_number": page_no,
                "source_file": source_file,
                "position": position,
            }

        # ───────── HEADINGS ─────────
        if "section_header" in label or label == "title":
            text = getattr(node, "text", "").strip()
            if text:
                current_section = text
                parsed_chunks.append({
                    "content": text,
                    "content_type": "text",
                    "metadata": _make_metadata("text", label),
                    "image_path": None
                })

        # ───────── TABLES ─────────
        elif "table" in label:
            table_text = ""

            if hasattr(node, "export_to_dataframe"):
                try:
                    df = node.export_to_dataframe()
                    if df is not None and not df.empty:
                        rows_text = []
                        headers = [str(c).strip() for c in df.columns]

                        for _, row in df.iterrows():
                            pairs = [
                                f"{h}: {str(v).strip()}"
                                for h, v in zip(headers, row)
                                if str(v).strip() not in ("", "nan", "None")
                            ]
                            if pairs:
                                rows_text.append("  |  ".join(pairs))

                        table_text = "\n".join(rows_text)
                except Exception:
                    pass

            if not table_text:
                table_text = getattr(node, "text", "")

            if table_text.strip():
                parsed_chunks.append({
                    "content": table_text.strip(),
                    "content_type": "table",
                    "metadata": _make_metadata("table", "table"),
                    "image_path": None
                })

        # ───────── 🔥 IMAGES ─────────
        elif "picture" in label or "figure" in label or label == "chart":
            caption = getattr(node, "text", "") or ""
            description = None
            image_path = None

            try:
                pil_img = None

                if hasattr(node, "get_image"):
                    pil_img = node.get_image(doc)

                elif hasattr(node, "image") and node.image:
                    pil_img = getattr(node.image, "pil_image", None)

                if pil_img:
                    # ✅ Save image locally
                    filename = f"{source_file}p{page_no}{len(parsed_chunks)}.png"
                    image_path = os.path.join(IMAGE_DIR, filename)

                    pil_img.save(image_path, format="PNG")

                    # ✅ Generate description
                    description = generate_image_description(pil_img)

            except Exception as e:
                description = f"Image processing failed: {str(e)}"

            content = description or caption.strip() or f"[Image on page {page_no}]"

            parsed_chunks.append({
                "content": content,
                "content_type": "image",
                "metadata": _make_metadata("image", "picture"),
                "image_path": image_path   # ✅ separate column
            })

        # ───────── TEXT ─────────
        else:
            text = getattr(node, "text", "")
            if text.strip():
                parsed_chunks.append({
                    "content": text.strip(),
                    "content_type": "text",
                    "metadata": _make_metadata("text", label),
                    "image_path": None
                })

    return parsed_chunks
