import os
import json
import asyncio
import contextlib
from collections import defaultdict
from aiohttp import ClientSession
from discord.ext import commands
from discord.interactions import Interaction
import tweepy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from discord import Embed
from dotenv import load_dotenv
import aiomysql

load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN = os.getenv("DISCORD_TOKEN")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# Setup Discord bot and Slash commands
bot = commands.Bot(command_prefix="!")
interaction_client = bot

# Setup Tweepy
auth = tweepy.OAuthHandler(os.environ["API_KEY"], os.environ["API_SECRET"])
auth.set_access_token(os.environ["ACCESS_TOKEN"], os.environ["ACCESS_SECRET"])
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


async def get_pool():
    return await aiomysql.create_pool(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        db=os.getenv("MYSQL_DATABASE"),
    )

pool = None

async def create_table(pool):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""CREATE TABLE IF NOT EXISTS twitter_channels (
                                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                                    twitter_account VARCHAR(255) NOT NULL,
                                    channel_id BIGINT NOT NULL,
                                    retweets BOOLEAN DEFAULT FALSE,
                                    replies BOOLEAN DEFAULT FALSE,
                                    media_only BOOLEAN DEFAULT FALSE,
                                    similarity_threshold FLOAT DEFAULT 0.5
                                )""")
            await conn.commit()

async def add_twitter_channel(pool, twitter_account, channel_id, retweets, replies, media_only, similarity_threshold):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("INSERT INTO twitter_channels (twitter_account, channel_id, retweets, replies, media_only, similarity_threshold) VALUES (%s, %s, %s, %s, %s, %s)",
                              (twitter_account, channel_id, retweets, replies, media_only, similarity_threshold))
            await conn.commit()

@bot.event
async def on_ready():
    global pool
    pool = await get_pool()
    await create_table(pool)
    print(f'{bot.user.name} has connected to Discord!')

@bot.command(name='start')
async def start(ctx, twitter_account: str, retweets: bool = False, replies: bool = False, media_only: bool = False, similarity_threshold: float = 0.5):
    await add_twitter_channel(pool, twitter_account, ctx.channel.id, retweets, replies, media_only, similarity_threshold)
    await ctx.send(f"Started streaming tweets from {twitter_account} to this channel with settings: retweets={retweets}, replies={replies}, media_only={media_only}, similarity_threshold={similarity_threshold}")
    # Start streaming tweets here

# Helper function to add a Twitter account to the database
async def add_twitter_account(twitter_handle: str, channel_id: int):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            # Get user info
            user = api.get_user(screen_name=twitter_handle)

            # Check if the account is already being monitored
            await cur.execute(
                "SELECT id FROM monitored_twitter_accounts WHERE twitter_id = %s",
                (user.id,)
            )
            account_id = await cur.fetchone()

            if account_id:
                account_id = account_id[0]
            else:
                # Add the new account to the monitored accounts
                await cur.execute(
                    "INSERT INTO monitored_twitter_accounts (twitter_id, twitter_handle) VALUES (%s, %s)",
                    (user.id, user.screen_name)
                )
                account_id = cur.lastrowid

            # Link the account to the specified channel
            await cur.execute(
                "INSERT INTO twitter_account_channels (twitter_account_id, discord_channel_id) VALUES (%s, %s)",
                (account_id, channel_id)
            )
            await conn.commit()


# Slash command to start monitoring a Twitter account
@interaction_client.command(name="start", description="Start monitoring a Twitter account.")
async def _start(ctx, twitter_handle: str):
    await add_twitter_account(twitter_handle, ctx.channel_id)
    await ctx.send(f"Now monitoring {twitter_handle} in this channel.")


# Function to remove similar headlines based on cosine similarity with TF-IDF
def remove_similar_headlines_tfidf(tweets, similarity_threshold=0.4):
    unique_tweets = []

    vectorizer = TfidfVectorizer(stop_words='english')
    for tweet in tweets:
        tweet_text = tweet.full_text.lower().strip()
        is_unique = True

        if unique_tweets:
            tweet_texts = [unique_tweet.full_text.lower().strip() for unique_tweet in unique_tweets]
            tweet_texts.append(tweet_text)
            tfidf_matrix = vectorizer.fit_transform(tweet_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            for similarity in similarity_matrix[-1][:-1]:
                if similarity >= similarity_threshold:
                    is_unique = False
                    break

        if is_unique:
            unique_tweets.append(tweet)

    return unique_tweets


# Function to stream tweets
async def stream_tweets():
    while True:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT * FROM monitored_twitter_accounts")
                accounts = await cur.fetchall()

                for account in accounts:
                    id, twitter_id, twitter_handle, last_tweet_id = account

                    # Get tweets since the last stored tweet ID
                    try:
                        if last_tweet_id:
                            tweets = api.user_timeline(
                                user_id=twitter_id,
                                since_id=last_tweet_id,
                                tweet_mode="extended",
                                include_rts=True,
                            )
                        else:
                            tweets = api.user_timeline(
                                user_id=twitter_id,
                                count=1,
                                tweet_mode="extended",
                                include_rts=True,
                            )

                        # Remove duplicate headlines
                        unique_tweets = remove_similar_headlines_tfidf(tweets)

                        # Send tweets to channels
                        for tweet in unique_tweets[::-1]:
                            await cur.execute(
                                "SELECT discord_channel_id FROM twitter_account_channels WHERE twitter_account_id = %s",
                                (id,)
                            )
                            channel_ids = await cur.fetchall()

                            for channel_id in channel_ids:
                                channel = bot.get_channel(channel_id[0])
                                if channel:
                                    await channel.send(
                                        f"**{tweet.user.screen_name}:** {tweet.full_text}\nhttps://twitter.com/i/web/status/{tweet.id}"
                                    )

                            # Update last_tweet_id
                            await cur.execute(
                                "UPDATE monitored_twitter_accounts SET last_tweet_id = %s WHERE id = %s",
                                (tweet.id, id)
                            )
                            await conn.commit()

                    except tweepy.TweepError as e:
                        print(f"Error fetching tweets for {twitter_handle}: {e}")


        # Wait before checking for new tweets
        await asyncio.sleep(30)

async def post_tweet_to_discord(client, tweet, channel_id):
    channel = client.get_channel(channel_id)
    embed = Embed(title=tweet['data']['text'], url=f"https://twitter.com/i/web/status/{tweet['data']['id']}", color=0x1DA1F2)

    # Handle media attachments
    if 'attachments' in tweet['data']:
        media_keys = tweet['data']['attachments'].get('media_keys', [])
        for media_key in media_keys:
            media = tweet['includes']['media'][media_keys.index(media_key)]
            media_url = media['url']

            if media['type'] in ['photo', 'animated_gif']:
                embed.set_image(url=media_url)
            elif media['type'] == 'video':
                # Get the best quality video variant
                best_variant = max(media['video_info']['variants'], key=lambda variant: variant.get('bitrate', 0))
                embed.description = f"[Video link]({best_variant['url']})"

    await channel.send(embed=embed)

# Start the tweet streaming background task
async def tweet_streamer():
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            await stream_tweets()
        except Exception as e:
            print(f"Error streaming tweets: {e}")
            await asyncio.sleep(30)

bot.loop.create_task(tweet_streamer())

if __name__ == "__main__":
    bot.run(os.environ["DISCORD_TOKEN"])
