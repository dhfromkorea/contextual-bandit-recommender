R6B - Yahoo! Front Page Today Module Us
Dataset: ydata-fp-td-clicks-v2_0

Yahoo! Front Page Today Module User Click Log Dataset, version 2.0

=====================================================================
This dataset is provided as part of the Yahoo! Webscope program, to be
used for approved non-commercial research purposes by recipients who 
have signed a Data Sharing Agreement with Yahoo!. This dataset is not
to be redistributed. No personally identifying information is available
in this dataset. More information about the Yahoo! Webscope program is
available at http://research.yahoo.com

=====================================================================

Full description:

This dataset contains 15 files, corresponding to 15 days in October 2011:

ydata-fp-td-clicks-v2_0.20111002.gz
ydata-fp-td-clicks-v2_0.20111003.gz
...
ydata-fp-td-clicks-v2_0.20111016.gz

Each line in the files corresponds to a user visit.  An example line is as
follows:

1317513291 id-560620 0 |user 1 9 11 13 23 16 18 17 19 15 43 14 39 30 66 50 27
104 20 |id-552077 |id-555224 |id-555528 |id-559744 |id-559855 |id-560290
|id-560518 |id-560620 |id-563115 |id-563582 |id-563643 |id-563787 |id-563846
|id-563938 |id-564335 |id-564418 |id-564604 |id-565364 |id-565479 |id-565515
|id-565533 |id-565561 |id-565589 |id-565648 |id-565747 |id-565822

which contains the following fields delimited with spaces:

    * timestamp: e.g., 1317513291
    * displayed_article_id: e.g., id-560620
    * user_click (0 for no-click and 1 for click): e.g., 0
    * string "|user" indicates the start of user features
    * features are 136-dimensional binary vectors; the IDs of nonzero features
are listed after the string "|user"
    * The pool of available articles for recommendation for each user visit is
the set of articles that appear in that line of data.  All user IDs (bcookies
in our data) are replaced by a common string "user".

Note that each user is associated with a 136-dimensional binary feature vector.
Features IDs take integer values in {1,2,...,136}.  Feature #1 is the constant
(always 1) feature, and features #2-136 correspond to other user information
such as age, gender, and behavior-targeting features, etc.  Some user features
are not present, since not all users logged in to Yahoo! when they visited the
front page.

A unique property of this data set is that the displayed article is chosen
uniformly at random from the candidate article pool.  Therefore, one can use an
unbiased offline evaluation method [Li et al., 2011] to compare bandit
algorithms in a reliable way.

Related publications for further information:
  * Evaluation methodology: http://dx.doi.org/10.1145/1935826.1935878
  * Reference performance: http://doi.acm.org/10.1145/1772690.1772758

=====================================================================
