import requests as rq
import pandas as pd
from tabulate import tabulate

def status_checker(url):
    """
    Checks a website's status and returns the data inside a Pandas DataFrame.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    try:
        response = rq.get(url, timeout=5)
        
        details = {
            "Metric": [
                "Target URL", 
                "Status Code", 
                "Status Reason", 
                "Response Time", 
                "Server Software", 
                "Content Type", 
                "Was Redirected"
            ],
            "Value": [
                response.url,
                int(response.status_code),
                response.reason,
                f"{response.elapsed.total_seconds():.3f} sec",
                response.headers.get('Server', 'Unknown'),
                response.headers.get('Content-Type', 'Unknown').split(';')[0],
                "Yes" if len(response.history) > 0 else "No"
            ]
        }
        return pd.DataFrame(details)

    except Exception as e:
        return pd.DataFrame({"Metric": ["Status", "Error"], "Value": ["Failed", str(e)]})



#print(tabulate(df_result, headers='keys', tablefmt='fancy_grid', showindex=False))