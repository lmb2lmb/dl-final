from numpy.core.arrayprint import format_float_scientific
import pandas as pd

# should be a lowercase list of banned words
banned_list = []
with open("badwords.txt", "r") as f:
    banned_list = f.read().splitlines()

jokes = pd.read_csv('shortjokes.csv')['Joke'].tolist()

filtered = []
for i, raw_joke in enumerate(jokes):
    if i % 10000 == 0:
        print(i)
    # check if any innaproriate words are contained within the joke
    bad = False
    for t in banned_list:
        if t in raw_joke.lower():
            bad=True
    if len(raw_joke) < 400 and not bad:
        filtered.append(raw_joke)

with open("filtered_jokes.txt", "w") as file:
    for j in filtered:
        file.write(j + "\n")

## strip punctuation
## save ?, ..., *, -, :, " as separate words
the_jokes = []
with open("filtered_jokes.txt", "w") as file:
    the_jokes = file.read().splitlines()






    

