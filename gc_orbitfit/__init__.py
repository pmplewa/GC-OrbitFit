import logging
import sys

# print to console even in a notebook session
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.__stdout__,
)
