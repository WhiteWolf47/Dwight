import discord
import os
import pandas as pd
import numpy as np
import tensorflow as tf

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

t_params = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
       'identity_hate']

model = tf.keras.models.load_model(r'toxicity.h5')
vmodel = tf.keras.models.load_model(r"vectorizer")
vectorizer = vmodel.layers[0]

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(t_params):
        text += '{}: {}\n'.format(col.upper(), results[0][idx] > 0.5)

    return text

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')

    if message.content.startswith('$score comment'):
        comment = message.content[15:]
        score = score_comment(comment)
        await message.channel.send(score)


#client.run(os.getenv("TOKEN"))
client.run("MTAyNTEzMzg3NDMwMjQyNzI2Nw.GOSm9T.ggCnGdrLVlW0kwpWiHZ92RLgyGK09jNOIc_HyA")
