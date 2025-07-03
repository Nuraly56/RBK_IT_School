import pandas as pd

data = {
    "Name": ["Ben 10", "Sponge bob", "Spider-Man", "Adventure time", "Avatar", "WITCH", "Regular show", "Danny Phantom", "Sonic X"],
    "Year": [2005, 1999, 1994, 2010, 2005, 2004, 2010, 2003, 2003],
    "TV_Channel": ["Cartoon Network", "Nickelodeon", "Jetix", "Cartoon Network", "Nickelodeon", "Jetix", "Cartoon Network", "Nickelodeon", "Jetix"],
    "IMDB": [7.6, 8.2, 8.4, 8.6, 9.3, 7.3, 8.6, 7.2, 6.3]
}


table = pd.DataFrame(data)
table.index = [f"Cartoon_{i+1}" for i in range(len(table))]
table["#"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
table.loc["Cartoon10"] = ["Naruto", 2002, "Jetix", 8.4, 10]


print(table)

