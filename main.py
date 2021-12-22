import asyncio
import logging

from decimal import Decimal
from tradingtools import assets
from tradingtools import utils
from tradingtools import visitors
from tradingtools import bot
from tradingtools import broker
from tradingtools import strategies
from tradingtools import data
from tradingtools.broker import exchanges, fillers
from tradingtools.data import dataloader

from tradingtools.utils import setup_signal_handlers


logging.basicConfig(level=logging.DEBUG)


config = bot.BotConfig(
    strategy__config=strategies.StrategyConfig(type=strategies.StrategyTypes.dummy),
    data_loader__config=dataloader.DataLoaderConfig(
        type=dataloader.DataLoaderTypes.dummy,
        pairs=["BTC-EUR", "ETH-EUR"],
        interval_length="2S",
    ),
    visitors__config=[],
    broker__config=broker.BrokerConfig(
        backtest=True,
        filler__config=fillers.FillStrategyConfig(type=fillers.FillerTypes.marketorder),
        exchange__config=exchanges.ExchangeConfig(
            type=exchanges.ExchangeTypes.dummy, backtest=True
        ),
    ),
)


bot = bot.Bot(config)

loop = asyncio.get_event_loop()

setup_signal_handlers(loop)

bot.start(loop)

loop.run_forever()
