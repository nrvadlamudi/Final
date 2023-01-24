# Machine Learning and NBA Playoff Success 
## An analysis of advanced regular season basketball statistics in predicting playoff wins

![image](https://user-images.githubusercontent.com/70409423/207753389-ecdb043f-a022-4deb-bcfa-ae93918faeb4.png)

### Data scraped from [https://www.basketball-reference.com/]

## Abstract
This repository contains a CatBoost regressor model that aims to predict playoff wins for each team in a given season based on their advanced regular season statistics. Advanced regular season statistics were determined and measured by [Basketball Reference](https://www.basketball-reference.com/) and data from the past 11 full NBA seasons to date (2021/2022 through 2010/2011) were used to train this model. Using CatBoost's Regressor model, I found that Wins, Losses, Pythagorean Wins (PW), Pythagorean Losses (L), Net Rating (Ntrtg), True Shooting Percentage (TS%), and Effective Field Goal Percentage (eFG%) were all signicant indicators of playoff success. Moving forward, it would be beneficial to see solely how advanced statistics are indicative of playoff success (i.e getting rid of Wins and Losses) and how a Classifier model could be better for grouping teams into more indicative clusters of types of playoff teams (non-playoff team, first round exit, championship contender, etc.) rather than numerically predicting the number of playoff wins.
