import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
import plotly.graph_objects as go

STOCK_LIST = ['NVDA', 'INTC', 'CRWD']
TRADING_DAYS = 252
DAYSPAN = 365

def get_stock_data(stocks, start_date, end_date):
    stock_data = yf.download(stocks, start=start_date, end=end_date)
    stock_data = stock_data['Close']
    stock_returns = stock_data.pct_change()

    return stock_returns.mean(), stock_returns.cov()

def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_returns = np.sum(mean_returns * weights) * TRADING_DAYS
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(TRADING_DAYS)

    return portfolio_returns, portfolio_std

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_returns - risk_free_rate) / p_std

# minimize negative SR
def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0, constraint_set=(0,1)):
    num_of_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for _ in range(num_of_assets))

    result = sco.minimize(negative_sharpe_ratio, num_of_assets * [1./num_of_assets],
                         args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def portfolio_variance(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

# minimize portfolio variance by altering weights/allocation of assets
def minimize_variance(mean_returns, cov_matrix, constraint_set=(0, 1)):
    num_of_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for _ in range(num_of_assets))

    result = sco.minimize(portfolio_variance, num_of_assets * [1. / num_of_assets],
                          args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return result

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=DAYSPAN)

mean_returns, cov_matrix = get_stock_data(STOCK_LIST, start_date, end_date)

def portfolio_returns(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]

def efficient_optimizer(mean_returns, cov_matrix, return_target, constraint_set=(0,1)):
    # for every target optimize portfolio for min variance
    num_of_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_returns(x, mean_returns, cov_matrix) - return_target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_of_assets))
    eff_opt_result = sco.minimize(portfolio_variance, num_of_assets * [1./num_of_assets],
                                  args=args, method='SLSQP', constraints=constraints, bounds=bounds)

    return eff_opt_result


def calculated_result(mean_returns, cov_matrix, risk_free_rate=0, constraint_set=(0,1)):
    # max sharpe ratio portfolio
    max_sr_portfolio = max_sharpe_ratio(mean_returns, cov_matrix)
    max_sr_returns, max_sr_std = portfolio_performance(max_sr_portfolio['x'], mean_returns, cov_matrix)
    max_sr_allocation = pd.DataFrame(max_sr_portfolio['x'], index=mean_returns.index, columns=['allocation'])
    max_sr_allocation.allocation = [round(i * 100, 0) for i in max_sr_allocation.allocation]

    # min variance portfolio
    min_var_portfolio = minimize_variance(mean_returns, cov_matrix)
    min_var_returns, min_var_std = portfolio_performance(min_var_portfolio['x'], mean_returns, cov_matrix)
    min_var_allocation = pd.DataFrame(min_var_portfolio['x'], index=mean_returns.index, columns=['allocation'])
    min_var_allocation.allocation = [round(i * 100, 0) for i in min_var_allocation.allocation]

    # efficient frontier
    efficient_list = []
    target_returns = np.linspace(min_var_returns, max_sr_returns, 20)
    for target in target_returns:
        efficient_list.append(efficient_optimizer(mean_returns, cov_matrix, target)['fun'])

    max_sr_returns, max_sr_std = round(max_sr_returns * 100, 2), round(max_sr_std * 100, 2)
    min_var_returns, min_var_std = round(min_var_returns * 100, 2), round(min_var_std * 100, 2)

    return max_sr_returns, max_sr_std, max_sr_allocation, min_var_returns, min_var_std, min_var_allocation, efficient_list, target_returns

#print(calculated_result(mean_returns, cov_matrix))

def ef_plot(mean_returns, cov_matrix, risk_free_rate=0, constraint_set=(0,1)):
    max_sr_returns, max_sr_std, max_sr_allocation, min_var_returns, min_var_std, min_var_allocation, efficient_list, target_returns = (
        calculated_result(mean_returns, cov_matrix, risk_free_rate, constraint_set))

    max_sharpe_ratio = go.Scatter(name='Maximum Sharpe Ratio',
                                  mode='markers',
                                  x=[max_sr_std],
                                  y=[max_sr_returns],
                                  marker=dict(color='red', size=14, line=dict(width=3, color='black')))

    min_var = go.Scatter(name='Maximum Volatility',
                                  mode='markers',
                                  x=[min_var_std],
                                  y=[min_var_returns],
                                  marker=dict(color='green', size=14, line=dict(width=3, color='black')))

    ef_curve = go.Scatter(name='Efficient Frontier',
                         mode='markers',
                         x=[round(ef_std * 100, 2) for ef_std in efficient_list],
                         y=[round(target * 100, 2) for target in target_returns],
                         line=dict(color='black', width=4, dash='dashdot'))

    data = [max_sharpe_ratio, min_var, ef_curve]

    layout = go.Layout(
        title = 'Portfolio optimization with the Efficient Frontier',
        yaxis = dict(title='Annualised return [%]'),
        xaxis = dict(title='Annualised volatility [%]'),
        showlegend=True,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal', bgcolor='#E2E2E2', bordercolor='black', borderwidth=2
        ),
        width = 800,
        height = 600
    )

    fig = go.Figure(data=data, layout=layout)
    return fig.show()

ef_plot(mean_returns, cov_matrix)
