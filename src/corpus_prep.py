import os
import re
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

NEWSGROUP_CATEGORIES = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
    "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
    "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
    "sci.space", "soc.religion.christian", "talk.politics.guns",
    "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc",
]

MIN_BODY_LENGTH = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "newsgroups"
EMBEDDING_BATCH_SIZE = 128