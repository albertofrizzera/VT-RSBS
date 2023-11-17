# -------------------------------------------------------------------------
# Author:   Alberto Frizzera, info@albertofrizzera.com
# Date:     11/10/2023
# -------------------------------------------------------------------------

import os
import sys
sys.path.append(os.path.join("."))
import json
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
from pdflatex import PDFLaTeX

from utils import time_convert, build_report


if __name__ == '__main__':
    
    initial_datetime = "20231027_09.13.25"

    params = json.load(open(os.path.join(os.path.dirname(__file__),"reports",initial_datetime,"report_"+initial_datetime+".json"), "r"))
    report_log = pd.read_csv(open(os.path.join(os.path.dirname(__file__),"reports",initial_datetime,"report_"+initial_datetime+".csv"), "rb"), index_col=0)

    build_report(params, report_log, initial_datetime, include_baseline=True)