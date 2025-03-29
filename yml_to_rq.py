import ruamel.yaml

# Load the YAML file
yaml = ruamel.yaml.YAML()
data = yaml.load(open('polor/condaenv.yml.yml'))

# Extract dependencies
requirements = []
for dep in data['dependencies']:
    if isinstance(dep, str):
        requirements.append(dep.split('=')[0])  # Extract package name
    elif isinstance(dep, dict):
        for pip_dep in dep.get('pip', []):
            requirements.append(pip_dep)

# Save to requirements.txt
with open('requirements.txt', 'w') as fp:
    for requirement in requirements:
        fp.write(requirement + '\n')

print("requirements.txt created successfully!")
