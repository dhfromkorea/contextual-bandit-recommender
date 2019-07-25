"""
Basic data model design:

User (one)---- (many) UserVisitEvent (one; displayed)-----------(many) Article
                         (one; considered for display)                 (many)
                           |                                             |
                           |                                             |
                         (many)                                        (one)
                      Shortlist           (many) ---------(one) ShortlistArticle
"""

from datetime import datetime

from sqlalchemy import (
        MetaData,
        create_engine,
        Column,
        Integer,
        Float,
        DateTime,
        ForeignKey,
)
from sqlalchemy.orm import (
        relationship,
        sessionmaker,
)
from sqlalchemy.ext.declarative import declarative_base

from sample_data import read_user_event, parse_uv_event, extract_data

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
    # trusting yahoo id as unique
    id = Column(Integer, primary_key=True, autoincrement=False)
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

        # start fresh for eval
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)

        # get table metadata
        meta = MetaData(bind=engine)
        meta.reflect()
        self._meta = meta

        Session = sessionmaker(bind=engine)
        self._session = Session()

        # if all tables exist
        if len(meta.sorted_tables) == 0:
            # create from scratch
            #Base.metadata.create_all(engine)
            self._shortlists = {}
            self._articles = set()

        else:
            q = self._session.query
            shortlists = q(Shortlist).all()
            shortlist_sets = {}
            for sl in shortlists:
                sl_arts = q(ShortlistArticle).filter_by(shortlist_id=sl.id)
                shortlist_sets[sl.id] = set([a.article_id for a in sl_arts])
            self._shortlists = shortlist_sets

            self._articles = set()
            articles = q(Article).all()
            for article in articles:
                self._articles.add(article.id)

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
            # check if article already exists
            if not art_id in self._articles:
                article = Article(id=art_id,
                                  feature_1=art_fs[0],
                                  feature_2=art_fs[1],
                                  feature_3=art_fs[2],
                                  feature_4=art_fs[3],
                                  feature_5=art_fs[4],
                                  feature_6=art_fs[5])
                self._session.add(article)
                self._articles.add(art_id)


        # art_ids belonging to current uv's shortlist
        cur_shortlist = set(uv["article"])
        past_shortlists = self._shortlists

        # check for shortlist
        if cur_shortlist in past_shortlists.values():
            # if shortlist_arts already exists
            for s_id, s_val in past_shortlists.items():
                if s_val == cur_shortlist:
                    shortlist_id = s_id
                    break
        else:
            # otherwise create a new shortlist
            shortlist = Shortlist()
            self._session.add(shortlist)
            self._session.flush()

            shortlist_id = shortlist.id
            self._shortlists[shortlist_id] = cur_shortlist

            # create shortlist article
            for art_id in cur_shortlist:
                shortlist_art = ShortlistArticle(article_id=art_id,
                                             shortlist_id=shortlist_id)
                self._session.add(shortlist_art)


        # create user visit event
        daid = uv["displayed_article_id"]

        date_time = datetime.fromtimestamp(int(uv["timestamp"]))

        uv_event = UserVisitEvent(user_id=user_id,
                                  displayed_article_id=daid,
                                  shortlist_id=shortlist_id,
                                  is_clicked=uv["is_clicked"],
                                  datetime=date_time)
        self._session.add(uv_event)


        # communicate changes so far to db
        # self._session.flush()



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


class DBHelperBulk(object):
    DB_NAME = "yahoo_news" + "_bulk"

    def __init__(self):
        """TODO: to be defined1. """

        #engine = create_engine("sqlite:///{}".format(self.DB_NAME), echo=True)
        engine = create_engine("sqlite:///{}".format(self.DB_NAME))
        self._engine = engine

        # start fresh for eval
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)


        # get table metadata
        meta = MetaData(bind=engine)
        meta.reflect()
        self._meta = meta

        Session = sessionmaker(bind=engine)
        self._session = Session()

        # if all tables exist
        if len(meta.sorted_tables) == 0:
            # create from scratch
            #Base.metadata.create_all(engine)
            self._shortlists = {}
            self._articles = set()
        else:
            q = self._session.query
            shortlists = q(Shortlist).all()
            shortlist_sets = {}
            for sl in shortlists:
                sl_arts = q(ShortlistArticle).filter_by(shortlist_id=sl.id)
                shortlist_sets[sl.id] = set([a.article_id for a in sl_arts])
            self._shortlists = shortlist_sets


            self._articles = set()
            articles = q(Article).all()
            for article in articles:
                self._articles.add(article.id)


    def write_user_event(self, parsed_uv_list):
        """
        """

        users = []
        # create users

        for uv in parsed_uv_list:
            fs = uv["user"]
            user = User(feature_1=fs[0],
                        feature_2=fs[1],
                        feature_3=fs[2],
                        feature_4=fs[3],
                        feature_5=fs[4],
                        feature_6=fs[5])

            users.append(user)

        self._session.bulk_save_objects(users, return_defaults=True)
        #self._session.add_all(users)
        #self._session.flush()
        # gives primary key


        # create articles
        articles = []
        for uv in parsed_uv_list:
            for art_id, art_fs in uv["article"].items():
                # check if article already exists
                if not art_id in self._articles:
                    # new article
                    article = Article(id=art_id,
                                      feature_1=art_fs[0],
                                      feature_2=art_fs[1],
                                      feature_3=art_fs[2],
                                      feature_4=art_fs[3],
                                      feature_5=art_fs[4],
                                      feature_6=art_fs[5])
                    articles.append(article)
                    self._articles.add(art_id)
        self._session.bulk_save_objects(articles)
        #self._session.add_all(articles)


        # check for shortlist
        shortlist_articles_raw = []

        shortlists = []
        for uv in parsed_uv_list:
            cur_shortlist = set(uv["article"])

            # pull existing shortlists
            past_shortlists = self._shortlists

            if cur_shortlist in past_shortlists.values():
                # if such shortlist already exists
                for s_id, s_val in past_shortlists.items():
                    if s_val == cur_shortlist:
                        shortlist_id = s_id
                        break
            else:
                # otherwise create a new shortlist
                shortlist = Shortlist()

                shortlists.append(shortlist)
                shortlist_articles_raw.append( (shortlist, cur_shortlist) )

        self._session.bulk_save_objects(shortlists, return_defaults=True)
        #self._session.add_all(shortlists)
        #self._session.flush()

        # create shortlist article
        shortlist_articles = []
        for shortlist, art_ids in shortlist_articles_raw:
            for art_id in art_ids:
                shortlist_art = ShortlistArticle(article_id=art_id,
                                                 shortlist_id=shortlist.id)

                shortlist_articles.append(shortlist_art)

            # update the existing shortlist pool
            self._shortlists[shortlist.id] = art_ids
        self._session.bulk_save_objects(shortlist_articles)
        #self._session.add_all(shortlist_articles)


        # create user visit event
        uv_events = []
        for uv, user, shortlist in zip(parsed_uv_list, users, shortlists):
            daid = uv["displayed_article_id"]

            date_time = datetime.fromtimestamp(int(uv["timestamp"]))

            uv_event = UserVisitEvent(user_id=user.id,
                                      displayed_article_id=daid,
                                      shortlist_id=shortlist.id,
                                      is_clicked=uv["is_clicked"],
                                      datetime=date_time)

            uv_events.append(uv_event)

        self._session.bulk_save_objects(uv_events)
        #self._session.add_all(uv_events)

        self._session.flush()


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


    def reset_all_tables(self):
        for tb in self._meta.sorted_tables:
            print("clear table: {}".format(tb))
            tb.drop(self._engine)


def main():
    """
    """

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
                print("{}th-batch 10000 samples written in {:.2f}s!".format(i, time.time() - start_t))
                # commit every now and then
                dbh._session.commit()
                start_t = time.time()

            uv_event_string = next(reader)
            uv = parse_uv_event(uv_event_string)

            if uv is None:
                # corrupted line; ignore
                continue

            dbh.write_user_event(uv)
            i += 1

            if i > 6 * 10**5:
                break

        except StopIteration:
            # end of file
            break

    dbh._session.commit()

def main_bulk():
    """
    there is no performance gain.

    due to the primary key retreival.

    """
    from sample_data import read_user_event, parse_uv_event

    dbh_bulk = DBHelperBulk()
    dbh_bulk.reset_all_data()

    reader = read_user_event()

    from pprint import pprint

    i = 0
    import time

    start_t = time.time()

    # for each file
    while True:
        try:
            uv_list = []
            for _ in range(10000):
                uv = parse_uv_event(next(reader))
                if not uv is None:
                    uv_list.append(uv)

            dbh_bulk.write_user_event(uv_list)
            print("{}th-batch 10000 samples written in {:.2f}s!".format(i, time.time() - start_t))
            start_t = time.time()

            i += 1

            if i > 60:
                break
        except StopIteration:
            # end of file
            break


    dbh_bulk._session.commit()


if __name__ == "__main__":
    """
    too slow
    """
    extract_data()

    main()
    #main_bulk()





