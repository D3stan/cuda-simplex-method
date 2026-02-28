from deepwiki_to_md import ContentExtractor, save_markdown_to_library

url = "https://deepwiki.com/shaowei-cai-group/Local-MIP/2.1-input-processing-pipeline"
base_dir = "./.deepwiki"  # equivalent to --path (optional)

extractor = ContentExtractor()
md = extractor.extract_from_url(url)

result = save_markdown_to_library(md, url, base_dir)
print("saved files:")
for p in result["saved_files"]:
    print(" -", p)
print("library index:", result["library_file"])  