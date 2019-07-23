"""
Basic data model design:

user (one)---- (many) user_visit_event (one; displayed)---(many)article
                         (one; considered for display)              (many)
                           |                                          |
                           |                                          |
                         (many)                                    (one)
                   shortlist (many) ---------(one)          shortlist_article
"""


from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy import MetaData, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker


Base = declarative_base()

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    feature_1 = Column(Float, nullable=False)
    feature_2 = Column(Float, nullable=False)
    feature_3 = Column(Float, nullable=False)
    feature_4 = Column(Float, nullable=False)
    feature_5 = Column(Float, nullable=False)
    feature_6 = Column(Float, nullable=False)
    user_visit_event = relationship("UserVisitEvent", back_populates="user")


class Article(Base):
    __tablename__ = "article"
    id = Column(Integer, primary_key=True)
    feature_1 = Column(Float, nullable=False)
    feature_2 = Column(Float, nullable=False)
    feature_3 = Column(Float, nullable=False)
    feature_4 = Column(Float, nullable=False)
    feature_5 = Column(Float, nullable=False)
    feature_6 = Column(Float, nullable=False)
    user_visit_event = relationship("UserVisitEvent", back_populates="article")


class ShortlistArticle(Base):
    __tablename__ = "shortlist_article"
    article_id = Column(Integer, ForeignKey("article.id"), primary_key=True)
    shortlist_id = Column(Integer, ForeignKey("shortlist.id"), primary_key=True)
    shortlist = relationship("Shortlist", back_populates="shortlist_article")


class Shortlist(Base):
    __tablename__ = "shortlist"
    id = Column(Integer, primary_key=True)
    shortlist_articles = relationship("ShortlistArticle", backref="shortlist")
    user_visit_events = relationship("UserVisitEvent", backref="shortlist")


class UserVisitEvent(Base):
    __tablename__ = "user_visit_event"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)

    user_id = Column(Integer, ForeignKey("user.id"))
    displayed_article_id = Column(Integer, ForeignKey("article.id"))
    portfolio_id = Column(Integer, ForeignKey("shortlist.id"))
    # how many different pools are there?



if __name__ == "__main__":
    DB_NAME = "yahoo_news"
    engine = create_engine("sqlite:///{}".format(DB_NAME), echo=True)

    metadata = MetaData(bind=engine)
    metadata.reflect()

    Session = sessionmaker(engine)
    session = Session()

    from sample_data import read_user_event, parse_uv_event

    reader = read_user_event()

    # for each file
    try:
        uv_event_string = next(reader)
        uv_event = parse_uv_event(uv_event_string)
    except StopIteration:
        # end of file
        pass
    import pdb;pdb.set_trace()


    # read file line by line
    # parse line
    # wrap into objects
    # write to db
    session.commit()
    # read from db
    # check










