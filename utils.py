# -------------------------------------------------------------------------
# Author:   Alberto Frizzera, info@albertofrizzera.com
# Date:     12/10/2023
# -------------------------------------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from pdflatex import PDFLaTeX


def time_convert(seconds):
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def draw_results_table(df, alignment="c", df_baseline=None):
    columns = pd.Series(df.columns.values).apply(lambda x: "\\rotatebox[origin=c]{90}{ \\textbf{"+x.replace("_","\_")+"} }")
    latex_table = "\\begin{tabular}{"+alignment*len(columns)+"""}
\hline
\hline
\\rowcolor[rgb]{0.835,0.835,0.835} """+" & ".join(columns)+""" \\\\"""
    if np.any(df_baseline):
        latex_table += """
\hline
\\rowcolor[rgb]{0.76,0.88,1} """+"\\n".join(df_baseline.iloc[0:1].to_latex(float_format="%.3f", index=False, header=False).split("\n")[3:-3])+"""
\hline
\\rowcolor[rgb]{0.84,1,0.82} """+"\\n".join(df_baseline.iloc[1:2].to_latex(float_format="%.3f", index=False, header=False).split("\n")[3:-3])
    latex_table += """
\hline
"""+"\\n".join(df.to_latex(float_format="%.3f", index=False, header=False).split("\n")[3:-3])+"""
\hline
\hline
\end{tabular}
"""
    return latex_table


def draw_params_table(params):
    df = pd.DataFrame([{"\\textbf{"+key.replace("_","\_")+"}": str(params[key]) for key in params.keys()}]).T
    df[0] = df[0].apply(lambda x: x.replace("_","\_"))
    latex_table = """\\begin{tabularx}{\columnwidth}{r|X}
"""+"\n".join(df.to_latex(header=False).split("\n")[3:-3])+"""
\end{tabularx}
"""
    return latex_table


def build_report(params, report_log, datetime, include_baseline=False):
    zero_shot_ids = np.where(report_log["dataset_type"]=="zero_shot")
    zero_shot_table = pd.DataFrame(columns=report_log.loc[zero_shot_ids]["dataset_name"].values)
    zero_shot_table.loc[0] = report_log.loc[zero_shot_ids]["accuracy"].values
    
    few_shot_ids = np.where(report_log["dataset_type"]=="few_shot")
    few_shot_table = pd.DataFrame(columns=report_log.loc[few_shot_ids]["dataset_name"].values)
    few_shot_table.loc[0] = report_log.loc[few_shot_ids]["accuracy"].values
    
    knn_ids = np.where(report_log["dataset_type"]=="knn")
    knn_table = pd.DataFrame(columns=report_log.loc[knn_ids]["dataset_name"].values)
    knn_table.loc[0] = report_log.loc[knn_ids]["accuracy"].values
    
    image_retrieval_ids = np.where(report_log["dataset_type"]=="image_retrieval")
    image_retrieval_table = pd.DataFrame(columns=report_log.loc[image_retrieval_ids]["dataset_name"].values)
    image_retrieval_table.loc[0] = report_log.loc[image_retrieval_ids]["accuracy"].values
    
    if include_baseline:
        report_log_CLIP_baseline = pd.read_csv(open(os.path.join(os.path.dirname(__file__),"reports","CLIP_baseline","report_CLIP_baseline.csv"), "rb"), index_col=0)
        report_log_CLIP_rsicd_v2_baseline = pd.read_csv(open(os.path.join(os.path.dirname(__file__),"reports","CLIP_rsicd_v2_baseline","report_CLIP_rsicd_v2_baseline.csv"), "rb"), index_col=0)
        
        report_log_CLIP_baseline = report_log_CLIP_baseline[report_log_CLIP_baseline["dataset_name"].isin(report_log["dataset_name"].values)].reset_index(drop=True)
        report_log_CLIP_rsicd_v2_baseline = report_log_CLIP_rsicd_v2_baseline[report_log_CLIP_rsicd_v2_baseline["dataset_name"].isin(report_log["dataset_name"].values)].reset_index(drop=True)
        
        assert np.all(report_log_CLIP_baseline["dataset_name"]==report_log_CLIP_rsicd_v2_baseline["dataset_name"]), "Error! Mismatch of the available datasets of the baseline models."
        zero_shot_ids = np.where(report_log_CLIP_baseline["dataset_type"]=="zero_shot")
        zero_shot_table_baseline = pd.DataFrame(columns=report_log_CLIP_baseline.loc[zero_shot_ids]["dataset_name"].values)
        zero_shot_table_baseline.loc[0] = report_log_CLIP_baseline.loc[zero_shot_ids]["accuracy"].values
        zero_shot_table_baseline.loc[1] = report_log_CLIP_rsicd_v2_baseline.loc[zero_shot_ids]["accuracy"].values
        
        few_shot_ids = np.where(report_log_CLIP_baseline["dataset_type"]=="few_shot")
        few_shot_table_baseline = pd.DataFrame(columns=report_log_CLIP_baseline.loc[few_shot_ids]["dataset_name"].values)
        few_shot_table_baseline.loc[0] = report_log_CLIP_baseline.loc[few_shot_ids]["accuracy"].values
        few_shot_table_baseline.loc[1] = report_log_CLIP_rsicd_v2_baseline.loc[few_shot_ids]["accuracy"].values
        
        knn_ids = np.where(report_log_CLIP_baseline["dataset_type"]=="knn")
        knn_table_baseline = pd.DataFrame(columns=report_log_CLIP_baseline.loc[knn_ids]["dataset_name"].values)
        knn_table_baseline.loc[0] = report_log_CLIP_baseline.loc[knn_ids]["accuracy"].values
        knn_table_baseline.loc[1] = report_log_CLIP_rsicd_v2_baseline.loc[knn_ids]["accuracy"].values
        
        image_retrieval_ids = np.where(report_log_CLIP_baseline["dataset_type"]=="image_retrieval")
        image_retrieval_table_baseline = pd.DataFrame(columns=report_log_CLIP_baseline.loc[image_retrieval_ids]["dataset_name"].values)
        image_retrieval_table_baseline.loc[0] = report_log_CLIP_baseline.loc[image_retrieval_ids]["accuracy"].values
        image_retrieval_table_baseline.loc[1] = report_log_CLIP_rsicd_v2_baseline.loc[image_retrieval_ids]["accuracy"].values
        
        baseline_legend = "\\noindent \n The baselines refer to the models: \colorbox{CLIPcolor}{\\textbf{CLIP}} and \colorbox{CLIPrsicdColor}{\\textbf{CLIP\_rsicd\_v2}}"
    else:
        zero_shot_table_baseline = None
        few_shot_table_baseline = None
        knn_table_baseline = None
        image_retrieval_table_baseline = None
        baseline_legend = ""
    
    environment = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__),"templates/")))
    template = environment.get_template("report.txt")
    context = {
        "datetime" : datetime.replace("_","\_"),
        "baseline_legend": baseline_legend,
        "model_table" : draw_params_table(params),
        "zero_shot_table" : draw_results_table(zero_shot_table, df_baseline=zero_shot_table_baseline),
        "few_shot_table" : draw_results_table(few_shot_table, df_baseline=few_shot_table_baseline),
        "knn_table" : draw_results_table(knn_table, df_baseline=knn_table_baseline),
        "image_retrieval_table" : draw_results_table(image_retrieval_table, df_baseline=image_retrieval_table_baseline)}
    if sys.platform == "linux" or sys.platform == "darwin":
        renderer_template = template.render(context).replace('\n','\r\n')
    else:
        renderer_template = template.render(context)
    with open(os.path.join(os.path.dirname(__file__),"reports",datetime,"report_"+datetime+".tex"), mode="w", encoding="utf-8") as results:
        results.write(renderer_template)
    pdfl = PDFLaTeX.from_texfile(os.path.join(os.path.dirname(__file__),"reports",datetime,"report_"+datetime+".tex"))
    pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=False, keep_log_file=False)
    with open(os.path.join(os.path.dirname(__file__),"reports",datetime,"report_"+datetime+".pdf"), 'wb') as f:
        f.write(pdf)
    with open(os.path.join(os.path.dirname(__file__),"reports",datetime,"report_"+datetime+".log"), 'wb') as f:
        f.write(log)