import pandas


data = pandas.read_csv('second/sepsis_file2.csv')
data_resistent = data[data['organism_resistence'] == 'R']
data_nonresistent = data[data['organism_resistence'] == 'S']
print("Quantidade de pacientes com bactérias resistentes", len(data_resistent))
print("Quantidade de pacientes com bactérias não resistentes", len(data_nonresistent))
print("Média de idade")
print('Resistente', data_resistent.mean()['age'], "max: ", data_resistent['age'].max(), "\tmin: ",data_resistent['age'].min())
print('Não resistente', data_nonresistent.mean()['age'], "max: ", data_nonresistent['age'].max(), "\tmin: ",data_nonresistent['age'].min())
print("Distribuição genero")
print("Resistente", data_resistent['GENDER'].value_counts().to_dict())
print("Não resistente", data_nonresistent['GENDER'].value_counts().to_dict())
# print("Etinicidade")
# print("Resistente", data_resistent['ethnicity'].value_counts().to_dict())
# print("Não resistente", data_nonresistent['ethnicity'].value_counts().to_dict())
print("Uso de vasopressores")
print("Resistente", data_resistent['vasopressor'].value_counts().to_dict())
print("Não resistente", data_nonresistent['vasopressor'].value_counts().to_dict())
print("Média de SOFA")
print("Resistente", data_resistent['sofa'].mean())
print("Não resistente", data_nonresistent['sofa'].mean())