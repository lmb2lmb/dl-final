import pandas as pd

# should be a lowercase list of banned words
banned_list = []
jokes = pd.read_csv('shortjokes.csv')['Joke'].tolist()

filtered = []
for raw_joke in jokes:
    # check if any innaproriate words are contained within the joke
    for t in raw_joke.lower():
        if t in banned_list:
            continue
    if len(raw_joke) > 400:
        continue
    filtered.append(raw_joke)

with open("filtered_jokes.txt", "w") as file:
    for j in filtered:
        file.write(j + "\n")







    

