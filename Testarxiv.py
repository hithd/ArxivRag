from langchain_community.document_loaders.arxiv import ArxivLoader

# 设置需要加载的论文ID列表
arxiv_ids = "1706.03762"

# 创建 ArxivLoader 实例
loader = ArxivLoader(arxiv_ids)

# 加载论文
documents = loader.load()

# 查看加载的文档信息
for doc in documents:
    print(doc.page_content)  # 论文的正文内容
    print(doc.metadata)  # 论文的元数据，比如标题、作者、摘要等
