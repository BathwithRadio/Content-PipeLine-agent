import os, re

from crewai.tools import tool
import dotenv
from firecrawl import FirecrawlApp, ScrapeOptions
import requests

API_KEY = os.getenv("FIRECRAWL_API_KEY")
dotenv.load_dotenv()

@tool
def web_search_tool(query: str):
    """
    Web Search Tool.
    Args:
        query: str
            The query to search the web for.
    Returns:
        A list of search results with the website content in Markdown format.
    """
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    response = app.search(
        query=query,
        limit=2,
        scrape_options=ScrapeOptions(
        formats=["markdown"]
        ),
    )

    if not response.success:
        return "Error using tool."

    cleaned_chunks = []

    for result in response.data:

        title = result["title"]
        url = result["url"]
        markdown = result["markdown"]

        cleaned = re.sub(r'\\+|\n+', '', markdown).strip()
        cleaned = re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)

        cleaned_result = {
        "title": title,
        "url": url,
        "markdown": cleaned,
        }

    cleaned_chunks.append(cleaned_result)

    return cleaned_chunks

# print(web_search_tool.run("React Native Jobs in Denmark"))
# def web_search_tool(query: str):
#     """Search jobs from web"""
    
#     app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    
#     response = app.search(
#         query=query,
#         limit=5,
#         scrape_options=ScrapeOptions(
#             formats=["markdown"],
#         ),
#     )
    
#     if response.success:
#         return "Error using tool."

#     cleaned_chunks = [ ]
    
#     for result in response.data:
#         title = result["title"]
#         url = result["url"]
#         markdown = result["markdown"]
        
#         cleaned = re.sub(r'\\+|\n+', '', markdown).strip()
        
#         cleaned_result = {
#             'title': title,
#             'url':url,
#             'markdown':cleaned,
            
#         }
        
#     cleaned_chunks.append(cleaned_result)
#     return cleaned_chunks
        
    
    
#for test
# print(web_search_tool.run("Firmware Jobs in Singapore"))