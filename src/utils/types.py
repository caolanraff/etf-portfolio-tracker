"""Type definitions."""
from datetime import date, datetime
from typing import Dict

import numpy as np
import pandas as pd
from typing_extensions import TypeAlias

DictFrame = Dict[str, pd.DataFrame]
Frame: TypeAlias = pd.DataFrame
Series: TypeAlias = pd.Series
Time: TypeAlias = date | datetime | pd.Timestamp | np.datetime64  # type: ignore
