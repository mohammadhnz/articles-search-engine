from elasticsearch_dsl import Document, Integer, Text, Keyword

from elasticsearch_dsl import connections, InnerDoc, Nested

connections.create_connection(
    hosts=['localhost'],
    timeout=20
)


class Author(InnerDoc):
    id = Integer()
    name = Keyword()


class Article(Document):
    id = Keyword()
    title = Text()
    authors = Nested(Author)
    abstract = Keyword()

    class Index:
        name = "article"
        settings = {"number_of_shards": 1, "number_of_replicas": 0}

    @classmethod
    def create(cls, id, title, authors, abstract):
        return Article(
            id=id,
            title=title,
            authors=[Author(id=auther['authorId'], name=auther['name']) for auther in authors],
            abstract=abstract
        )
