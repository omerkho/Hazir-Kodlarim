bet <- read.csv("turkey.csv")
season <- grep("2016", bet$Season)
last_season <- bet[season, ]
library(package = "ggplot2")
stsl <- grep(1, last_season$tier)
stsl16 <- last_season[stsl, ]
season_matrix <- ggplot(last_season, aes(x = home, y = visitor, color = result)) + geom_point(size = 2)
season_matrix <- season_matrix + theme(axis.text = element_text(angle=90))

season_matrix <- season_matrix+ ggtitle("Stsl 16/17 results matrix")
season_matrix <- season_matrix + scale_color_manual(name = "Result", labels = c("Away win","Draw","Home win"), values = c("#E69F00","#999999","#56B4E9"))
season_matrix + labs(x = "home team", y = "away team")
