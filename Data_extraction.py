from string import ascii_uppercase
from itertools import pairwise
from openpyxl import load_workbook
import pandas as pd
import os
from datetime import date

def get_next_code(code):
    '''
    returns next column letter given existing one
    input: 'AA', output: 'AB'
    input: 'AB', output: 'AC'
    '''
    letter_map = {a:b for a,b in pairwise(ascii_uppercase)}
    code = list(code)
    i = -1
    while True:
        if code[i] == 'Z':
            code[i] = 'A'
            i -= 1
            if abs(i) > len(code):
                return 'A' + ''.join(code)
        else:
            code[i] = letter_map[code[i]]
            return ''.join(code)
        
        
# def snap_table(sheet, topleft, num_columns, num_rows):
#     '''
#     input:
#         sheet: the Excel sheet we want
#         topleft: coordinates of topleft coordinate eg. 'B2'
#         num_columns: number of columns in table
#         num_rows: number of rows in table
#     output:
#         pandas dataframe representing table
#     '''
#     try:
#         import re
#         col = re.findall('[a-zA-Z]+', topleft)[0]
#         num = int(re.findall('[0-9]+', topleft)[0])

#         columns = [col]
#         for i in range(num_columns-1):
#             columns.append(get_next_code(columns[-1]))
#         numbers = [n for n in range(num, num+num_rows)]

#         data = []
#         for n in numbers:
#             row = []
#             for c in columns:
#                 code = c + str(n)
#                 row.append(sheet[code].value)
#             data.append(row)
#         return pd.DataFrame(data[1:], columns=data[0])
#     except Exception as e:
#         return pd.DataFrame()
def snap_table(sheet, topleft, num_columns, num_rows):
    '''
    input:
        sheet: the Excel sheet we want
        topleft: coordinates of topleft coordinate eg. 'B2'
        num_columns: number of columns in table
        num_rows: number of rows in table
    output:
        pandas dataframe representing table
    '''
    try:
        import re
        col = re.findall('[a-zA-Z]+', topleft)[0]
        num = int(re.findall('[0-9]+', topleft)[0])

        columns = [col]
        for i in range(num_columns-1):
            columns.append(get_next_code(columns[-1]))
        numbers = [n for n in range(num, num+num_rows)]

        data = []
        for n in numbers:
            row = []
            for c in columns:
                code = c + str(n)
                row.append(sheet[code].value or sheet[code].value)
            data.append(row)
        return pd.DataFrame(data[1:], columns=data[0])
    except Exception as e:
        return pd.DataFrame()

    
# load path to excel file + select your excel sheet
workbook = load_workbook('Modèle_Belfaa_updated_linearized_1303_simul_noQ.xlsx') 
sheet1 = workbook['latest inputs']
sheet2=workbook['Indices']
sheet3=workbook['Simulation']
df_price=snap_table(sheet1, 'B10',91, 3)
df_prod=snap_table(sheet1, 'B17',45, 16)
df_chargesvar=snap_table(sheet1, 'B38',8, 16)
df_plantation=snap_table(sheet2, 'J52',3, 11)
df_month_index=snap_table(sheet2, 'L3',3, 25)
df_sim=snap_table(sheet3, 'B22',3, 25)

# Define the folder path (replace 'your_username' with your actual username)
folder_path = ".\Data"

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)  # exist_ok avoids errors if folder exists

# Export DataFrames with full path
df_price.to_csv(os.path.join(folder_path, "Prices.csv"), index=False)
df_chargesvar.to_csv(os.path.join(folder_path, "Charges_var.csv"), index=False)
df_prod.to_csv(os.path.join(folder_path, "Production.csv"), index=False)
df_plantation.to_csv(os.path.join(folder_path, "plantation.csv"), index=False)
df_month_index.to_csv(os.path.join(folder_path, "month_index.csv"), index=False)
df_sim.to_csv(os.path.join(folder_path, "Simulation.csv"), index=False)

print(f"DataFrames saved to folder: {folder_path}")