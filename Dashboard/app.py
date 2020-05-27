from flask import Flask, render_template, request
import pandas as pd
from model_plots import predictions, data_loan, show_map, show_pie, show_bar, show_scatter


# translate flask to python object
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    # -------- state and state code
    states = pd.read_csv('states.csv')
    zipped = zip(list(states['State']),list(states['Code']))
    # -------- prediction
    if request.method == 'POST':
        data = request.form
        data = data.to_dict()
        data['UrbanRural'] = int(data['UrbanRural'])
        data['NewExist'] = int(data['NewExist'])
        data['FranchiseCode'] = int(data['FranchiseCode'])
        data['RealEstate'] = int(data['RealEstate'])
        data['NoEmp'] = int(data['NoEmp'])
        data['Term'] = int(data['Term'])
        data['DisbursementGross'] = int(data['DisbursementGross'])
        data['SBA_Appv'] = int(data['SBA_Appv'])
        data['LowDoc'] = int(data['LowDoc'])
        hasil = predictions(data)
        return render_template('result.html',data=data,prediction_status=hasil[0],prediction_loan=hasil[1],warna1=hasil[2][0],warna2=hasil[2][1],warna3=hasil[2][2])
    return render_template('prediction.html',data_state=zipped)


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/data')
def data():
    datatable = data_loan()
    datamap = show_map()
    return render_template('data.html',datatable=datatable,datamap=datamap)

@app.route('/data2')
def data2():
    datatable = data_loan()
    datapie = show_pie()
    return render_template('data2.html',datatable=datatable,datapie=datapie)

@app.route('/data3')
def data3():
    datatable = data_loan()
    databar = show_bar()
    return render_template('data3.html',datatable=datatable,databar=databar)

@app.route('/data4')
def data4():
    datatable = data_loan()
    datascatter = show_scatter()
    return render_template('data4.html',datatable=datatable,datascatter=datascatter)

# @app.route('/evaluation')
# def evaluation():
#     data = show_scatter()
#     return render_template('evaluation.html',data=data)

if __name__ == '__main__':
    app.run(debug=True,port=2300)