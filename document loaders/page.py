from langchain_community.document_loaders import WebBaseLoader

url = "https://www.samsung.com/in/smartphones/galaxy-s26-ultra/buy/?cid=in_pd_search_google_all-products-brandshop-rlsa_ongoing_all-products-brandshop-26-rlsa-gd2c-pfm_text_text-search-ad_1ur-560042l-2026&gad_source=1&gad_campaignid=23617950935&gbraid=0AAAAADm5iQVcronaJOIPOD_e9ax776dlj&gclid=CjwKCAjw2rrQBhBuEiwAarLWHblnP1FVNXvMPzGG4xgjdBTD7vBvC_hAnsWg73M4j9BRsBF4rPmx9xoCPfYQAvD_BwE"

data = WebBaseLoader(url)

docs = data.load()

print(len(docs))