{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.utils import shuffle\n",
    "from palmerpenguins import load_penguins"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>species</th>\n",
       "      <th>island</th>\n",
       "      <th>bill_length_mm</th>\n",
       "      <th>bill_depth_mm</th>\n",
       "      <th>flipper_length_mm</th>\n",
       "      <th>body_mass_g</th>\n",
       "      <th>sex</th>\n",
       "      <th>year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Torgersen</td>\n",
       "      <td>39.1</td>\n",
       "      <td>18.7</td>\n",
       "      <td>181.0</td>\n",
       "      <td>3750.0</td>\n",
       "      <td>male</td>\n",
       "      <td>2007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Torgersen</td>\n",
       "      <td>39.5</td>\n",
       "      <td>17.4</td>\n",
       "      <td>186.0</td>\n",
       "      <td>3800.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Torgersen</td>\n",
       "      <td>40.3</td>\n",
       "      <td>18.0</td>\n",
       "      <td>195.0</td>\n",
       "      <td>3250.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Torgersen</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Torgersen</td>\n",
       "      <td>36.7</td>\n",
       "      <td>19.3</td>\n",
       "      <td>193.0</td>\n",
       "      <td>3450.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2007</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  \\\n",
       "0  Adelie  Torgersen            39.1           18.7              181.0   \n",
       "1  Adelie  Torgersen            39.5           17.4              186.0   \n",
       "2  Adelie  Torgersen            40.3           18.0              195.0   \n",
       "3  Adelie  Torgersen             NaN            NaN                NaN   \n",
       "4  Adelie  Torgersen            36.7           19.3              193.0   \n",
       "\n",
       "   body_mass_g     sex  year  \n",
       "0       3750.0    male  2007  \n",
       "1       3800.0  female  2007  \n",
       "2       3250.0  female  2007  \n",
       "3          NaN     NaN  2007  \n",
       "4       3450.0  female  2007  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "penguins = load_penguins()\n",
    "penguins.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(344, 8)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "penguins.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "species               0\n",
      "island                0\n",
      "bill_length_mm        2\n",
      "bill_depth_mm         2\n",
      "flipper_length_mm     2\n",
      "body_mass_g           2\n",
      "sex                  11\n",
      "year                  0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(penguins.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "penguins.dropna(axis=0, how='any', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(333, 8)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "penguins.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [],
   "source": [
    "penguins_shuffle=shuffle(penguins, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>species</th>\n",
       "      <th>island</th>\n",
       "      <th>bill_length_mm</th>\n",
       "      <th>bill_depth_mm</th>\n",
       "      <th>flipper_length_mm</th>\n",
       "      <th>body_mass_g</th>\n",
       "      <th>sex</th>\n",
       "      <th>year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Dream</td>\n",
       "      <td>39.5</td>\n",
       "      <td>16.7</td>\n",
       "      <td>178.0</td>\n",
       "      <td>3250.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>320</th>\n",
       "      <td>Chinstrap</td>\n",
       "      <td>Dream</td>\n",
       "      <td>50.9</td>\n",
       "      <td>17.9</td>\n",
       "      <td>196.0</td>\n",
       "      <td>3675.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2009</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>79</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Torgersen</td>\n",
       "      <td>42.1</td>\n",
       "      <td>19.1</td>\n",
       "      <td>195.0</td>\n",
       "      <td>4000.0</td>\n",
       "      <td>male</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>202</th>\n",
       "      <td>Gentoo</td>\n",
       "      <td>Biscoe</td>\n",
       "      <td>46.6</td>\n",
       "      <td>14.2</td>\n",
       "      <td>210.0</td>\n",
       "      <td>4850.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>63</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Biscoe</td>\n",
       "      <td>41.1</td>\n",
       "      <td>18.2</td>\n",
       "      <td>192.0</td>\n",
       "      <td>4050.0</td>\n",
       "      <td>male</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  \\\n",
       "30      Adelie      Dream            39.5           16.7              178.0   \n",
       "320  Chinstrap      Dream            50.9           17.9              196.0   \n",
       "79      Adelie  Torgersen            42.1           19.1              195.0   \n",
       "202     Gentoo     Biscoe            46.6           14.2              210.0   \n",
       "63      Adelie     Biscoe            41.1           18.2              192.0   \n",
       "\n",
       "     body_mass_g     sex  year  \n",
       "30        3250.0  female  2007  \n",
       "320       3675.0  female  2009  \n",
       "79        4000.0    male  2008  \n",
       "202       4850.0  female  2008  \n",
       "63        4050.0    male  2008  "
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "penguins_shuffle.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>species</th>\n",
       "      <th>island</th>\n",
       "      <th>bill_length_mm</th>\n",
       "      <th>bill_depth_mm</th>\n",
       "      <th>flipper_length_mm</th>\n",
       "      <th>body_mass_g</th>\n",
       "      <th>sex</th>\n",
       "      <th>year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td></td>\n",
       "      <td>Dream</td>\n",
       "      <td>39.5</td>\n",
       "      <td>16.7</td>\n",
       "      <td>178.0</td>\n",
       "      <td>3250.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>320</th>\n",
       "      <td></td>\n",
       "      <td>Dream</td>\n",
       "      <td>50.9</td>\n",
       "      <td>17.9</td>\n",
       "      <td>196.0</td>\n",
       "      <td>3675.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2009</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>79</th>\n",
       "      <td></td>\n",
       "      <td>Torgersen</td>\n",
       "      <td>42.1</td>\n",
       "      <td>19.1</td>\n",
       "      <td>195.0</td>\n",
       "      <td>4000.0</td>\n",
       "      <td>male</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>202</th>\n",
       "      <td></td>\n",
       "      <td>Biscoe</td>\n",
       "      <td>46.6</td>\n",
       "      <td>14.2</td>\n",
       "      <td>210.0</td>\n",
       "      <td>4850.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>63</th>\n",
       "      <td></td>\n",
       "      <td>Biscoe</td>\n",
       "      <td>41.1</td>\n",
       "      <td>18.2</td>\n",
       "      <td>192.0</td>\n",
       "      <td>4050.0</td>\n",
       "      <td>male</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  \\\n",
       "30               Dream            39.5           16.7              178.0   \n",
       "320              Dream            50.9           17.9              196.0   \n",
       "79           Torgersen            42.1           19.1              195.0   \n",
       "202             Biscoe            46.6           14.2              210.0   \n",
       "63              Biscoe            41.1           18.2              192.0   \n",
       "\n",
       "     body_mass_g     sex  year  \n",
       "30        3250.0  female  2007  \n",
       "320       3675.0  female  2009  \n",
       "79        4000.0    male  2008  \n",
       "202       4850.0  female  2008  \n",
       "63        4050.0    male  2008  "
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = penguins_shuffle.copy()\n",
    "df.iloc[:30, df.columns.get_loc(\"species\")] = \"\"\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop([\"species\", \"island\", \"sex\", \"year\"], axis=1)\n",
    "y = df[\"species\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(random_state=42)"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Adelie', 'Adelie', 'Chinstrap', 'Adelie', 'Adelie', 'Gentoo',\n",
       "       'Adelie', 'Chinstrap', 'Gentoo', 'Adelie', 'Gentoo', 'Gentoo',\n",
       "       'Gentoo', 'Adelie', 'Gentoo', 'Gentoo', 'Gentoo', 'Gentoo',\n",
       "       'Adelie', 'Gentoo', 'Gentoo', 'Chinstrap', 'Gentoo', 'Adelie',\n",
       "       'Chinstrap', 'Adelie', 'Gentoo', 'Adelie', 'Chinstrap', 'Adelie'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Acurácia: 86.67%\n"
     ]
    }
   ],
   "source": [
    "acertos = (y_pred == y_test).sum()\n",
    "total_previsoes = len(y_pred)\n",
    "\n",
    "acuracia = acertos / total_previsoes * 100\n",
    "\n",
    "print(f'Acurácia: {acuracia:.2f}%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Gustavo\\AppData\\Local\\Temp\\ipykernel_12792\\1022874111.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df['species'][:30] = y_pred\n"
     ]
    }
   ],
   "source": [
    "df['species'][:30] = y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>species</th>\n",
       "      <th>island</th>\n",
       "      <th>bill_length_mm</th>\n",
       "      <th>bill_depth_mm</th>\n",
       "      <th>flipper_length_mm</th>\n",
       "      <th>body_mass_g</th>\n",
       "      <th>sex</th>\n",
       "      <th>year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Dream</td>\n",
       "      <td>39.5</td>\n",
       "      <td>16.7</td>\n",
       "      <td>178.0</td>\n",
       "      <td>3250.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>320</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Dream</td>\n",
       "      <td>50.9</td>\n",
       "      <td>17.9</td>\n",
       "      <td>196.0</td>\n",
       "      <td>3675.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2009</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>79</th>\n",
       "      <td>Chinstrap</td>\n",
       "      <td>Torgersen</td>\n",
       "      <td>42.1</td>\n",
       "      <td>19.1</td>\n",
       "      <td>195.0</td>\n",
       "      <td>4000.0</td>\n",
       "      <td>male</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>202</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Biscoe</td>\n",
       "      <td>46.6</td>\n",
       "      <td>14.2</td>\n",
       "      <td>210.0</td>\n",
       "      <td>4850.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>63</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Biscoe</td>\n",
       "      <td>41.1</td>\n",
       "      <td>18.2</td>\n",
       "      <td>192.0</td>\n",
       "      <td>4050.0</td>\n",
       "      <td>male</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>195</th>\n",
       "      <td>Gentoo</td>\n",
       "      <td>Biscoe</td>\n",
       "      <td>49.6</td>\n",
       "      <td>15.0</td>\n",
       "      <td>216.0</td>\n",
       "      <td>4750.0</td>\n",
       "      <td>male</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>77</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Torgersen</td>\n",
       "      <td>37.2</td>\n",
       "      <td>19.4</td>\n",
       "      <td>184.0</td>\n",
       "      <td>3900.0</td>\n",
       "      <td>male</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Biscoe</td>\n",
       "      <td>39.7</td>\n",
       "      <td>17.7</td>\n",
       "      <td>193.0</td>\n",
       "      <td>3200.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2009</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>281</th>\n",
       "      <td>Chinstrap</td>\n",
       "      <td>Dream</td>\n",
       "      <td>45.2</td>\n",
       "      <td>17.8</td>\n",
       "      <td>198.0</td>\n",
       "      <td>3950.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>108</th>\n",
       "      <td>Adelie</td>\n",
       "      <td>Biscoe</td>\n",
       "      <td>38.1</td>\n",
       "      <td>17.0</td>\n",
       "      <td>181.0</td>\n",
       "      <td>3175.0</td>\n",
       "      <td>female</td>\n",
       "      <td>2009</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>333 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  \\\n",
       "30      Adelie      Dream            39.5           16.7              178.0   \n",
       "320     Adelie      Dream            50.9           17.9              196.0   \n",
       "79   Chinstrap  Torgersen            42.1           19.1              195.0   \n",
       "202     Adelie     Biscoe            46.6           14.2              210.0   \n",
       "63      Adelie     Biscoe            41.1           18.2              192.0   \n",
       "..         ...        ...             ...            ...                ...   \n",
       "195     Gentoo     Biscoe            49.6           15.0              216.0   \n",
       "77      Adelie  Torgersen            37.2           19.4              184.0   \n",
       "112     Adelie     Biscoe            39.7           17.7              193.0   \n",
       "281  Chinstrap      Dream            45.2           17.8              198.0   \n",
       "108     Adelie     Biscoe            38.1           17.0              181.0   \n",
       "\n",
       "     body_mass_g     sex  year  \n",
       "30        3250.0  female  2007  \n",
       "320       3675.0  female  2009  \n",
       "79        4000.0    male  2008  \n",
       "202       4850.0  female  2008  \n",
       "63        4050.0    male  2008  \n",
       "..           ...     ...   ...  \n",
       "195       4750.0    male  2008  \n",
       "77        3900.0    male  2008  \n",
       "112       3200.0  female  2009  \n",
       "281       3950.0  female  2007  \n",
       "108       3175.0  female  2009  \n",
       "\n",
       "[333 rows x 8 columns]"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "30        Adelie\n",
       "320       Adelie\n",
       "79     Chinstrap\n",
       "202       Adelie\n",
       "63        Adelie\n",
       "307       Gentoo\n",
       "292       Adelie\n",
       "187    Chinstrap\n",
       "219       Gentoo\n",
       "204       Adelie\n",
       "81        Gentoo\n",
       "14        Gentoo\n",
       "330       Gentoo\n",
       "132       Adelie\n",
       "276       Gentoo\n",
       "138       Gentoo\n",
       "120       Gentoo\n",
       "152       Gentoo\n",
       "82        Adelie\n",
       "286       Gentoo\n",
       "115       Gentoo\n",
       "143    Chinstrap\n",
       "326       Gentoo\n",
       "206       Adelie\n",
       "6      Chinstrap\n",
       "116       Adelie\n",
       "272       Gentoo\n",
       "334       Adelie\n",
       "169    Chinstrap\n",
       "333       Adelie\n",
       "Name: species, dtype: object"
      ]
     },
     "execution_count": 109,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"species\"][:30]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "30        Adelie\n",
       "320    Chinstrap\n",
       "79        Adelie\n",
       "202       Gentoo\n",
       "63        Adelie\n",
       "307    Chinstrap\n",
       "292    Chinstrap\n",
       "187       Gentoo\n",
       "219       Gentoo\n",
       "204       Gentoo\n",
       "81        Adelie\n",
       "14        Adelie\n",
       "330    Chinstrap\n",
       "132       Adelie\n",
       "276    Chinstrap\n",
       "138       Adelie\n",
       "120       Adelie\n",
       "152       Gentoo\n",
       "82        Adelie\n",
       "286    Chinstrap\n",
       "115       Adelie\n",
       "143       Adelie\n",
       "326    Chinstrap\n",
       "206       Gentoo\n",
       "6         Adelie\n",
       "116       Adelie\n",
       "272       Gentoo\n",
       "334    Chinstrap\n",
       "169       Gentoo\n",
       "333    Chinstrap\n",
       "Name: species, dtype: object"
      ]
     },
     "execution_count": 110,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "penguins_shuffle[\"species\"][:30]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.2 (tags/v3.9.2:1a79785, Feb 19 2021, 13:44:55) [MSC v.1928 64 bit (AMD64)]"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "8b7cd9fa445b4fb759c484121e60e22797bcff94abd34c0cb93db981118a194f"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
