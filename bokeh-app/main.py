from bokeh.models import ColumnDataSource, Whisker, Band
from bokeh.plotting import figure, show
from bokeh.models.widgets import Select
from bokeh.layouts import column, layout
from bokeh.io import curdoc
import pandas as pd
import numpy as np
from bokeh.models import HoverTool

df = pd.read_csv('./plotdata24.csv')
pvals = pd.read_csv('./fdr24.csv')
df=df.dropna()
pvals=pvals.dropna()

predictor_dict = {'donorparity': 'Donor parity','idbloodgroupcat': 'ABO identical transfusion','meandonationtime': 'Time of donation','meandonorage': 'Age of Donor','meandonorhb': 'Donor Hb','meandonorsex': 'Donor sex','meanstoragetime': 'Storage time (days)','meanweekday': 'Weekday of donation','numdoncat': 'Donors prior number of donations','timesincecat': 'Time since donors previous donation'}
label_dict = {'ALAT': 'ALT','ALB': 'Albumin','ALP': 'ALP','APTT': 'aPTT','ASAT': 'AST','BASOF': 'Basophiles','BE': 'Base Excess','BILI': 'Bilirubin','BILI_K': 'Conjugated bilirubin','BLAST': 'Blast cells','CA': 'Calcium','CA_F': 'Free Calcium','CL': 'Chloride','CO2': 'Carbon Dioxide','COHB': 'CO-Hb','CRP': 'CRP','EGFR': 'eGFR','EOSINO': 'Eosinophile count','ERYTRO': 'Erythrocyte count','ERYTROBL': 'Erythroblasts','EVF': 'EVF','FE': 'Iron','FERRITIN': 'Ferritin','FIB': 'Fibrinogen','GLUKOS': 'Glucose','GT': 'Glutamyl transferase','HAPTO': 'Haptoglobin','HB': 'Hemoglobin','HBA1C': 'HbA1c','HCT': 'Hematocrit','INR': 'INR','K': 'Potassium','KREA': 'Creatinine','LAKTAT': 'Lactate','LD': 'Lactate dehydrogenase','LPK': 'Leukocyte count','LYMF': 'Lymphocyte count','MCH': 'Mean corpuscular  hemoglobin','MCHC': 'Mean corpuscular  hemoglobin concentration','MCV': 'Mean corpuscular volume','META': 'Metamyelocyte count','METHB': 'Methemoglobin','MONO': 'Monocyte count','MYELO': 'Myelocyte count','NA': 'Sodium','NEUTRO': 'Neutrophile count','NTPROBNP': 'NT-ProBNP','OSMO': 'Osmolality','PCO2': 'PaCO2','PH': 'pH','PO2': 'PaO2','RET': 'Reticulocyte count','STDBIK': 'Standard bicarbonate','TPK': 'Platelet count','TRI': 'Triglycerides','TROP_I': 'Troponin I','TROP_T': 'Troponin T'}
inv_predictor_dict = {v: k for k, v in predictor_dict.items()}
inv_label_dict = {v: k for k, v in label_dict.items()}


def create_figure(current_predictor, current_label):
    #current_predictor = predictor_select.value
    #current_label = label_select.value
    df_current = df[(df['label'] == current_label) & (df['predictor'] == current_predictor)]
    current_fpval =  float(pvals[(pvals['label'] == current_label) & (pvals['predictor'] == current_predictor)]['ProbF'].iloc[0])
    current_fdrp =  float(pvals[(pvals['label'] == current_label) & (pvals['predictor'] == current_predictor)]['fdr_p'].iloc[0])

    p = figure(width=1200, height=800, 
        title='Dose-response plot of the association between %s and %s. Crude p-value=%s, FDR-adjusted p-value=%s' % (label_dict[current_label], predictor_dict[current_predictor],np.format_float_scientific(current_fpval,precision=1),np.format_float_scientific(current_fdrp,precision=1)))


    if current_predictor in ['donorparity', 'idbloodgroupcat', 'meandonorsex', 'meanweekday', 'numdoncat', 'timesincecat']:
        p.circle(df_current['predictorvalue'], df_current['predicted'], size=10)
        p.add_layout(
            Whisker(source=ColumnDataSource(df_current), base="predictorvalue", upper="upper", lower="lower")
        )
        p.xaxis.ticker = np.arange(int(min(df_current['predictorvalue'])), int(max(df_current['predictorvalue'])) + 1)  # ensure x-axis ticks are integers
    else:
        source = ColumnDataSource(df_current)
        band = Band(base='predictorvalue', lower='lower', upper='upper', source=source, level='underlay',
                    fill_alpha=0.2, line_width=1, line_color='black')
        p.add_layout(band)
        p.line(df_current['predictorvalue'], df_current['predicted'], line_width=2)
    
    p.yaxis.axis_label = "Delta %s (95%% CI)" % label_dict[current_label]
    p.xaxis.axis_label = predictor_dict[current_predictor]
    
    p.title.text_font_size = '12pt'
    p.title.align = 'left'
    
    x_start = df_current['predictorvalue'].min()
    x_end = df_current['predictorvalue'].max()
    y_start = df_current[['lower', 'predicted', 'upper']].min().min()
    y_end = df_current[['lower', 'predicted', 'upper']].max().max()

    # Add 10% padding to ranges
    x_padding = (x_end - x_start) * 0.05
    y_padding = (y_end - y_start) * 0.05

    p.x_range.start = x_start - x_padding
    p.x_range.end = x_end + x_padding
    p.y_range.start = y_start - y_padding
    p.y_range.end = y_end + y_padding
    
    # Update x and y range to ensure that the plot fits the data
    #p.x_range.start = df_current['predictorvalue'].min()
    #p.x_range.end = df_current['predictorvalue'].max()
    #p.y_range.start = df_current[['lower', 'pred', 'upper']].min().min()
    #p.y_range.end = df_current[['lower', 'pred', 'upper']].max().max()
    
    hover = HoverTool(tooltips=[
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("desc", "@desc"),
        ])
    p.add_tools(hover)


    return p

def update(attr, old, new):
    current_predictor = inv_predictor_dict[predictor_select.value]
    current_label = inv_label_dict[label_select.value]
    df_current = df[(df['label'] == current_label) & (df['predictor'] == current_predictor)]
    layout.children[1] = create_figure(current_predictor, current_label)

predictor_select = Select(title='Select Donor-Donation-Component characteristic', value='Donor Hb', options=[predictor_dict[i] for i in df['predictor'].unique().tolist()])
predictor_select.on_change('value', update)

label_select = Select(title='Select laboratory test', value='Hemoglobin', options=[label_dict[i] for i in df['label'].unique().tolist()])
label_select.on_change('value', update)

# predictor_select = Select(title='Select Donor-Donation-Component characteristic', value='meandonorhb', options=df['predictor'].unique().tolist())
# predictor_select.on_change('value', update)

# label_select = Select(title='Select laboratory test', value='HB', options=df['label'].unique().tolist())
# label_select.on_change('value', update)

controls = column(predictor_select, label_select)

initial_predictor = inv_predictor_dict[predictor_select.value]
initial_label = inv_label_dict[label_select.value]

layout = column(controls, create_figure(initial_predictor, initial_label))

curdoc().add_root(layout)