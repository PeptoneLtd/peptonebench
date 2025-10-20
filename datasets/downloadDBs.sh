if [ "$(basename "$PWD")" != "datasets" ]
then
  echo "Please run this script from the 'datasets' directory"
  exit 1
fi
wget https://zenodo.org/record/17306061/files/PeptoneDBs.tar.gz
tar -xvzf PeptoneDBs.tar.gz && rm PeptoneDBs.tar.gz
