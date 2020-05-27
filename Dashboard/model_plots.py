import pickle
import pandas as pd
import plotly.express as px
import plotly
import plotly.graph_objects as go
import plotly.express as px
import json

model = pickle.load(open('LightGBM.sav','rb'))
scaler_term = pickle.load(open('scalernormal_term.sav','rb'))
scaler_emp = pickle.load(open('scalernormal_emp.sav','rb'))
scaler_gross = pickle.load(open('scalernormal_gross.sav','rb'))
scaler_sba = pickle.load(open('scalernormal_sba.sav','rb'))


## Using import csv ---------------------
# ---- option 1 using SQL
# engine = create_engine("mysql+mysqlconnector://root:passwordagung@localhost/FinalProject?host=localhost?port=3306")
# connection = engine.connect()
# result = connection.execute('SELECT * from FinalProject.final_dataframe_revision').fetchall()
# df_main = pd.DataFrame(result, columns = result[0].keys())
# df_main.drop(['Unnamed: 0','LoanNr_ChkDgt',"GrAppv"],axis = 1, inplace = True)

# ------ option 2 import using pandas
df_main = pd.read_csv('df_fixnew.csv')
df_main = df_main.drop(['Unnamed: 0','LoanNr_ChkDgt',"GrAppv"],axis=1)
# df_dashboard = df_main.copy()

# -----------------------------

cols = ['Term',
 'NoEmp',
 'NewExist',
 'FranchiseCode',
 'UrbanRural',
 'LowDoc',
 'DisbursementGross',
 'SBA_Appv',
 'RealEstate',
 'NAICS_Administrative & Support / Waste Management / Remediation Services',
 'NAICS_Agriculture / Forestry / Fishing / Hunting',
 'NAICS_Arts / Entertainment / Recreation',
 'NAICS_Construction',
 'NAICS_Educational',
 'NAICS_Finance / Insurance',
 'NAICS_Health Care / Social Assistance',
 'NAICS_Information',
 'NAICS_Management of Companies and Enterprises',
 'NAICS_Manufacturing',
 'NAICS_Mining / Quarrying / Oil&Gas Extraction',
 'NAICS_Other Services (except public admin)',
 'NAICS_Proffesional / Scientific / Tech.Service',
 'NAICS_Public Administration',
 'NAICS_Real Estate / Rental / Leasing',
 'NAICS_Retail trade',
 'NAICS_Transportation / Warehousing',
 'NAICS_Utilities',
 'NAICS_Wholesale trade',
 'State_AL',
 'State_AR',
 'State_AZ',
 'State_CA',
 'State_CO',
 'State_CT',
 'State_DC',
 'State_DE',
 'State_FL',
 'State_GA',
 'State_HI',
 'State_IA',
 'State_ID',
 'State_IL',
 'State_IN',
 'State_KS',
 'State_KY',
 'State_LA',
 'State_MA',
 'State_MD',
 'State_ME',
 'State_MI',
 'State_MN',
 'State_MO',
 'State_MS',
 'State_MT',
 'State_NC',
 'State_ND',
 'State_NE',
 'State_NH',
 'State_NJ',
 'State_NM',
 'State_NV',
 'State_NY',
 'State_OH',
 'State_OK',
 'State_OR',
 'State_PA',
 'State_RI',
 'State_SC',
 'State_SD',
 'State_TN',
 'State_TX',
 'State_UT',
 'State_VA',
 'State_VT',
 'State_WA',
 'State_WI',
 'State_WV',
 'State_WY']


def predictions(data):
    output = []
    df = pd.DataFrame(data,index=[0])
    df = pd.get_dummies(df,columns=['NAICS','State'],drop_first=True)
    df = df.reindex(columns=cols,fill_value=0)
    df['Term'] = scaler_term.transform(df[['Term']])
    df['NoEmp'] = scaler_emp.transform(df[['NoEmp']])
    df['DisbursementGross'] = scaler_gross.transform(df[['DisbursementGross']])
    df['SBA_Appv'] = scaler_sba.transform(df[['SBA_Appv']])
    pred_proba = model.predict_proba(df)
    # Adjust threshold
    pred_th = []
    for item in pred_proba[:,0]:
        if item > 0.24 :
            pred_th.append(0)
        else:
            pred_th.append(1)       
    pred = pred_th[0]
    # label
    if pred == 0 :
        output.append('HIGH RISK')
        output.append('CHARGE OFF')
        output.append([250,0,0])
    else :
        output.append('LOW RISK')
        output.append('PAID IN FULL')
        output.append([0,0,250])
    return output

def data_loan():
    data_loan = df_main.sample(30,random_state=7)
    return data_loan

def show_map():
    # -- import csv that contain state name
    states = pd.read_csv('states.csv')
    # -- percentage of CHGOFF based on state
    gb1 = df_main[df_main['MIS_Status'] == 'CHGOFF'].groupby('State').count()[['MIS_Status']].reset_index()
    gb2 = df_main[df_main['MIS_Status'] == 'P I F'].groupby('State').count()[['MIS_Status']].reset_index()
    gb_state = pd.merge(gb1,gb2,on='State')
    gb_state.rename(columns={'MIS_Status_x':'CHGOFF','MIS_Status_y':'P I F'},inplace=True)
    def percent(x):
        a = x['CHGOFF'] / (x['P I F'] + x['CHGOFF'])
        return round(a,3)
    gb_state['Percent'] = gb_state.apply(percent,axis=1)
    # -- Input State Name based on state code
    def apply_name_states(x) :
        return states[states['Code'] == x['State']]['State'].values[0]
    gb_state['StateName'] = gb_state.apply(apply_name_states,axis=1)
    df_state_MIS = gb_state.copy()
    # -- Chloropleth
    fig = go.Figure(data=go.Choropleth(
        locations=df_state_MIS['State'], # Spatial coordinates
        z = df_state_MIS['Percent'].astype(float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'spectral',
        text = df_state_MIS['StateName'],
        colorbar_title = "Percentage",
    ))
    fig.update_layout(
        title_text = 'CGHOFF Percentage per State',
        geo_scope='usa', # limite map scope to USA
    )
    final_fig = fig
    fig_json = json.dumps(final_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json

def show_pie():
    pie = px.pie(df_main,'MIS_Status',hole=0.5,title='Status of Loan')
    fig_json = json.dumps(pie, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json

def show_bar():
    df_gb = df_main.groupby(['ApprovalFY','MIS_Status']).count()[['State']].reset_index()
    fig = px.bar(df_gb,x='ApprovalFY',y='State',color='MIS_Status',barmode='group',color_discrete_sequence=["orangered","royalblue"],title='Loan Status from 2000 - 2014')
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json

def show_scatter():
    fig = px.scatter(df_main,x='NoEmp',y='DisbursementGross',color='MIS_Status',title='Disbursement Gross vs Numbers of Employees')
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json