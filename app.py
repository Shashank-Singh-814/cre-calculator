from flask import Flask, render_template, request
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    graph_html = None
    result = None
    X = None
    error = None

    if request.method == 'POST':
        mode = request.form['mode']

        if mode == 'conversion':
            try:
                k = float(request.form['k'])
                X = float(request.form['X'])
                C_A0 = 1.0
                if not 0 < X < 1:
                    raise ValueError("Conversion X must be between 0 and 1.")
                t_conversion = -np.log(1 - X) / k
                result = round(t_conversion, 4)
            except Exception as e:
                error = str(e)

        else:
            try:
                k = float(request.form['k'])
                V = float(request.form['V'])
                t0 = float(request.form['t0'])
                tf = float(request.form['tf'])
                steps = int(request.form['steps'])

                num_reactants = int(request.form['num_reactants'])
                num_products = int(request.form['num_products'])

                t_eval = np.linspace(t0, tf, steps)
                C0 = {}
                stoich = {}

                for i in range(1, num_reactants + 1):
                    C0[f'A{i}'] = float(request.form.get(f'C_A{i}', 0.0))
                    stoich[f'A{i}'] = float(request.form.get(f'nu_A{i}', 1))

                for i in range(1, num_products + 1):
                    C0[f'B{i}'] = 0.0
                    stoich[f'B{i}'] = float(request.form.get(f'nu_B{i}', 1))

                def dCdt(t, y):
                    local_C = {f'A{i+1}': y[i] for i in range(num_reactants)}
                    rate = k
                    for i in range(1, num_reactants + 1):
                        rate *= local_C[f'A{i}'] ** stoich[f'A{i}']
                    dydt = []
                    for i in range(1, num_reactants + 1):
                        dydt.append(-stoich[f'A{i}'] * rate)
                    for i in range(1, num_products + 1):
                        dydt.append(stoich[f'B{i}'] * rate)
                    return dydt

                y0 = [C0[f'A{i}'] for i in range(1, num_reactants + 1)] + [C0[f'B{i}'] for i in range(1, num_products + 1)]
                sol = solve_ivp(dCdt, (t0, tf), y0, t_eval=t_eval)

                fig = go.Figure()
                species = list(C0.keys())
                for i, label in enumerate(species):
                    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[i], mode='lines+markers', name=label))

                fig.update_layout(
                    title='Batch Reactor Simulation',
                    xaxis_title='Time (s)',
                    yaxis_title='Concentration (mol/L)',
                    legend=dict(x=0.01, y=0.99),
                    template='plotly_white',
                    hovermode='x unified'
                )
                graph_html = pio.to_html(fig, full_html=False)

                data = {'Time': sol.t}
                for i, label in enumerate(species):
                    data[label] = sol.y[i]
                df = pd.DataFrame(data)
                csv_io = BytesIO()
                df.to_csv(csv_io, index=False)
                csv_io.seek(0)
                with open('static/batch_output.csv', 'wb') as f:
                    f.write(csv_io.read())

            except Exception as e:
                error = str(e)

    return render_template(
        'batch.html',
        graph_html=graph_html,
        result=result,
        X=X,
        error=error,
        csv_available=True if graph_html else False
    )

@app.route('/cstr')
def cstr():
    return "<h1 style='text-align: center;'>CSTR Module Coming Soon</h1>"

@app.route('/pfr')
def pfr():
    return "<h1 style='text-align: center;'>PFR Module Coming Soon</h1>"

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

