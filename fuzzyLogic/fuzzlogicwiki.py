import wikipediaapi
import pdfkit




def print_sections(sections, level=0):
        for s in sections:
                print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text[0:40]))
                print_sections(s.sections, level + 1)
def print_categories(page):
        categories = page.categories
        for title in sorted(categories.keys()):
            print("%s: %s" % (title, categories[title]))

def print_categorymembers(categorymembers, level=0, max_level=2):
        category_list = []
        for c in categorymembers.values():
            print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
            if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                print_categorymembers(c.categorymembers, level=level + 1, max_level=max_level)
                category_list.append(c.categorymembers)
        return category_list

wiki_wiki = wikipediaapi.Wikipedia('FuzzyLogicWiki.py/1.0 (aaron.stevenson408@gmail.com)','en')
category_title = 'Category:Fuzzy_logic'
category = wiki_wiki.page(category_title)
# print_sections(category.sections)
categoory_list = print_categorymembers(category.categorymembers)
print(categoory_list)
# def extract_content_from_category(category_content):
#     # You need to define your own logic to extract relevant content from the category page
#     # This will depend on the structure of the Wikipedia category page you are working with

#     # For demonstration purposes, let's assume we want to keep everything after the "== Pages in category ==" section
#     pages_section_start = category_content.find("== Pages in category ==")
#     if pages_section_start != -1:
#         return category_content[pages_section_start:].strip()
#     return None

# def save_to_pdf(content, filename='output.pdf'):
#     # Convert HTML content to PDF using pdfkit
#     pdfkit.from_string(content, filename)

# if __name__ == "__main__":
#     # Specify a user agent in the headers to be polite to Wikipedia

#     # Replace 'Category:Fuzzy_logic' with the actual Wikipedia category title you want to scrape
#     category_title = 'Category:Fuzzy_logic'
    
#     # Download Wikipedia category page content
#     category_content = download_wikipedia_category(category_title)
    
#     if category_content:
#         # Extract relevant content (you need to customize this based on the structure of the category page)
#         relevant_content = extract_content_from_category(category_content)
#         if relevant_content:
#             print("Relevant Content:")
#             print(relevant_content)
        
#         # Save the category content to a PDF file
#         save_to_pdf(relevant_content)
#         print("PDF created successfully.")
# import wikipediaapi
# from bs4 import BeautifulSoup
# import weasyprint

# def download_wikipedia_category(category_title):
#     # Specify a user agent in the headers to be polite to Wikipedia
#     headers = {'User-Agent': 'YourAppName/1.0 (your@email.com)'}
    
#     wiki_wiki = wikipediaapi.Wikipedia('en')
#     category = wiki_wiki.page(category_title)
    
#     # Check if the category exists
#     if category.exists():
#         return category.text
#     else:
#         print(f"Error: Wikipedia category '{category_title}' not found.")
#         return None

# def extract_content_from_category(category_content):
#     # You need to define your own logic to extract relevant content from the category page
#     # This will depend on the structure of the Wikipedia category page you are working with

#     # For demonstration purposes, let's assume we want to keep everything after the "== Pages in category ==" section
#     pages_section_start = category_content.find("== Pages in category ==")
#     if pages_section_start != -1:
#         return category_content[pages_section_start:].strip()
#     return None

# def save_to_pdf(content, filename='output.pdf'):
#     # Use weasyprint to convert HTML content to PDF
#     with open(filename, 'wb') as pdf_file:
#         weasyprint.HTML(string=content).write_pdf(pdf_file)

# if __name__ == "__main__":
#     # Replace 'Category:Fuzzy_logic' with the actual Wikipedia category title you want to scrape
#     category_title = 'Category:Fuzzy_logic'
    
#     # Download Wikipedia category page content
#     category_content = download_wikipedia_category(category_title)
    
#     if category_content:
#         # Extract relevant content (you need to customize this based on the structure of the category page)
#         relevant_content = extract_content_from_category(category_content)
#         if relevant_content:
#             print("Relevant Content:")
#             print(relevant_content)
        
#         # Save the category content to a PDF file using weasyprint
#         save_to_pdf(relevant_content)
#         print("PDF created successfully.")
