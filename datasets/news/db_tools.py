"""
Basic data model design:

user (one)---- (many) user_visit_event (one; displayed)---(many)article
                         (one; considered for display)              (many)
                           |                                          |
                           |                                          |
                         (many)                                    (one)
                   shortlist (many) ---------(one)          shortlist_article
"""

from datetime import datetime


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
    uv_events = relationship("UserVisitEvent", backref="user")


class Article(Base):
    __tablename__ = "article"
    id = Column(Integer, primary_key=True)
    yahoo_id = Column(Integer, nullable=False)
    feature_1 = Column(Float, nullable=False)
    feature_2 = Column(Float, nullable=False)
    feature_3 = Column(Float, nullable=False)
    feature_4 = Column(Float, nullable=False)
    feature_5 = Column(Float, nullable=False)
    feature_6 = Column(Float, nullable=False)
    shortlists = relationship("ShortlistArticle", backref="article")


class ShortlistArticle(Base):
    __tablename__ = "shortlist_article"
    article_id = Column(Integer, ForeignKey("article.id"), primary_key=True)
    shortlist_id = Column(Integer, ForeignKey("shortlist.id"), primary_key=True)


class Shortlist(Base):
    __tablename__ = "shortlist"
    id = Column(Integer, primary_key=True)
    # ref to many
    shortlist_articles = relationship("ShortlistArticle", backref="shortlist")
    user_visit_events = relationship("UserVisitEvent", backref="shortlist")


class UserVisitEvent(Base):
    __tablename__ = "user_visit_event"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    displayed_article_id = Column(Integer, ForeignKey("article.id"))
    shortlist_id = Column(Integer, ForeignKey("shortlist.id"))
    datetime = Column(DateTime)
    is_clicked = Column(Integer)


class DBHelper(object):
    DB_NAME = "yahoo_news"

    def __init__(self):
        """TODO: to be defined1. """

        #engine = create_engine("sqlite:///{}".format(self.DB_NAME), echo=True)
        engine = create_engine("sqlite:///{}".format(self.DB_NAME))
        self._engine = engine


        # get table metadata
        meta = MetaData(bind=engine)
        meta.reflect()
        self._meta = meta

        Session = sessionmaker(bind=engine)
        self._session = Session()

        # if all tables exist
        if len(meta.sorted_tables) == 0:
            # create from scratch
            Base.metadata.create_all(engine)
            self._shortlists = {}
        else:
            q = self._session.query
            shortlists = q(Shortlist).all()
            shortlist_sets = {}
            for sl in shortlists:
                sl_arts = q(ShortlistArticle).filter_by(shortlist_id=sl.id)
                shortlist_sets[sl.id] = set([a.article_id for a in sl_arts])
            self._shortlists = shortlist_sets




    def write_user_event(self, parsed_user_event):
        """
        """
        uv = parsed_user_event

        # create user
        fs = uv["user"]
        user = User(feature_1=fs[0],
                    feature_2=fs[1],
                    feature_3=fs[2],
                    feature_4=fs[3],
                    feature_5=fs[4],
                    feature_6=fs[5])
        self._session.add(user)
        # gives primary key
        self._session.flush()
        user_id = user.id


        # create article
        for art_id, art_fs in uv["article"].items():
            article = Article(yahoo_id=art_id,
                              feature_1=art_fs[0],
                              feature_2=art_fs[1],
                              feature_3=art_fs[2],
                              feature_4=art_fs[3],
                              feature_5=art_fs[4],
                              feature_6=art_fs[5])
            self._session.add(article)


        # check for shortlist
        shortlist_arts = set(uv["article"])

        shortlists = self._shortlists

        if shortlist_arts in shortlists.values():
            # if shortlist_arts already exists
            for s_id, s_val in shortlists.items():
                if s_val == shortlist_arts:
                    shortlist_id = s_id
                    break
        else:
            # otherwise create a new shortlist
            shortlist = Shortlist()
            self._session.add(shortlist)
            self._session.flush()
            shortlist_id = shortlist.id


            # create shortlist article
            shortlist_arts = set()
            for art_id in uv["article"]:
                short_art = ShortlistArticle(article_id=art_id,
                                             shortlist_id=shortlist_id)
                self._session.add(short_art)
                shortlist_arts.add(art_id)

            self._shortlists[shortlist_id] = shortlist_arts



        # create user visit event
        daid = uv["displayed_article_id"]

        date_time = datetime.fromtimestamp(int(uv["timestamp"]))

        UserVisitEvent(user_id=user_id,
                       displayed_article_id=daid,
                       shortlist_id=shortlist_id,
                       is_clicked=uv["is_clicked"],
                       datetime=date_time)



    @property
    def users(self):
        q = self._session.query
        users = q(User).all()
        return users


    @property
    def articles(self):
        q = self._session.query
        articles = q(Article).all()
        return articles


    @property
    def events(self):
        q = self._session.query
        uv_events = q(UserVisitEvent).all()
        return uv_events


    def reset_all_data(self):
        for tb in self._meta.sorted_tables:
            print("clear table: {}".format(tb))
            self._session.execute(tb.delete())
        self._session.commit()



if __name__ == "__main__":
    """
    too slow
    """


    from sample_data import read_user_event, parse_uv_event

    dbh = DBHelper()

    dbh.reset_all_data()

    reader = read_user_event()

    from pprint import pprint

    i = 0
    import time

    start_t = time.time()

    # for each file
    while True:
        try:
            if i % 10000 == 0:
                print("{}th write done in {}s!".format(i, time.time() - start_t))
                start_t = time.time()
                dbh._session.flush()

            uv_event_string = next(reader)
            uv = parse_uv_event(uv_event_string)
            dbh.write_user_event(uv)

            i += 1

        except StopIteration:
            # end of file
            break


    dbh._session.commit()

    #for art in dbh.articles:
    #    pprint(art.id)
    #for et in dbh.events:
    #    pprint(et.id)
    #for u in dbh.users:
    #    pprint(u.id)


    # read file line by line
    # parse line
    # wrap into objects
    # write to db
    # read from db
    # check










