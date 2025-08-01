import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import json
from pathlib import Path
import re

class RegressionModel:
    def __init__(self, data_path=None):
        self.models = {}
        self.features = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']
        self.target = 'Total_Sales'
        
        if data_path is None:
            self.data_path = Path(__file__).resolve().parents[3] / 'AI/data/sales_data.csv'
        else:
            self.data_path = Path(data_path)

        print(f"--- 데이터 로드 시작: {self.data_path} ---")
        try:
            try:
                self.raw_df = pd.read_csv(self.data_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("UTF-8 디코딩 실패. latin1으로 재시도합니다.")
                self.raw_df = pd.read_csv(self.data_path, encoding='latin1')
            
            self.monthly_data = self._preprocess_and_aggregate(self.raw_df.copy())
            print("--- 데이터 로드 및 전처리 완료 ---")
            
        except FileNotFoundError:
            print(f"!!! 에러: 데이터 파일을 찾을 수 없습니다: {self.data_path}")
            self.raw_df = None
            self.monthly_data = None
        except Exception as e:
            print(f"!!! 데이터 로드 중 에러 발생: {e}")
            self.raw_df = None
            self.monthly_data = None

    def _preprocess_and_aggregate(self, df):
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce', dayfirst=True)
        df.dropna(subset=['Order Date'], inplace=True)
        df['Month'] = df['Order Date'].dt.to_period('M')
        
        agg_df = df.groupby(['Month', 'Sub-Category', 'Product Name']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum',
            'Discount': 'mean'
        }).reset_index()
        
        agg_df.rename(columns={
            'Sales': 'Total_Sales',
            'Profit': 'Total_Profit',
            'Quantity': 'Total_Quantity',
            'Discount': 'Avg_Discount'
        }, inplace=True)
        
        return agg_df

    def _create_features(self, df_group):
        df_group = df_group.sort_values('Month').copy()
        for lag in [1, 2, 3]:
            df_group[f'lag_{lag}'] = df_group[self.target].shift(lag)
        df_group['rolling_mean_3'] = df_group[self.target].shift(1).rolling(window=3).mean()
        return df_group.dropna()

    def train(self, df, target_column='Total_Sales'):
        df_featured = self._create_features(df)
        
        if len(df_featured) < 2:
            return None

        X = df_featured[self.features]
        y = df_featured[target_column]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=1)
        model.fit(X, y)
        return model

    def predict(self, product_input: str) -> str:
        if self.monthly_data is None:
            return json.dumps({"type": "error", "reason": "데이터가 로드되지 않았습니다."})
            
        match = re.search(r"([^,]+)", product_input)
        product_name = match.group(1).strip() if match else product_input.strip()

        MIN_MONTHS_FOR_TRAINING = 6
        
        product_data = self.monthly_data[self.monthly_data['Product Name'] == product_name]
        
        df_to_use = None
        result_type = ""
        name_for_model = ""
        reason = None

        if len(product_data) >= MIN_MONTHS_FOR_TRAINING:
            print(f"--- '{product_name}'의 고유 데이터({len(product_data)}개월)로 예측합니다. ---")
            df_to_use = product_data
            result_type = "product_specific"
            name_for_model = product_name
        else:
            sub_category_rows = self.raw_df[self.raw_df['Product Name'] == product_name]
            if not sub_category_rows.empty:
                sub_category_name = sub_category_rows['Sub-Category'].iloc[0]
                reason = f"'{product_name}'의 데이터가 부족({len(product_data)}개월), 상위 카테고리 '{sub_category_name}' 모델로 예측합니다."
                print(reason)
                
                category_data = self.monthly_data[self.monthly_data['Sub-Category'] == sub_category_name]
                df_to_use = category_data.groupby('Month').agg({'Total_Sales': 'sum'}).reset_index()

                result_type = "sub_category_fallback"
                name_for_model = sub_category_name
            else:
                return json.dumps({"type": "error", "reason": f"'{product_name}'에 대한 데이터가 부족하고, 상위 카테고리도 찾을 수 없습니다."})

        if df_to_use is None or len(df_to_use) < 4:
            return json.dumps({"type": "error", "reason": f"'{name_for_model}'에 대한 학습 데이터가 피처 생성을 하기에 부족합니다."})

        model_key = f"{name_for_model}_Total_Sales"
        if model_key not in self.models:
            self.models[model_key] = self.train(df_to_use)
        
        model = self.models[model_key]
        if model is None:
            return json.dumps({"type": "error", "reason": f"'{name_for_model}' 모델 학습에 실패했습니다 (데이터 부족)." })

        df_featured_for_pred = self._create_features(df_to_use.copy())
        if df_featured_for_pred.empty:
             return json.dumps({"type": "error", "reason": f"'{name_for_model}'에 대한 예측용 피처 생성에 실패했습니다."})

        last_month_features = df_featured_for_pred.iloc[-1][self.features].values.reshape(1, -1)
        prediction = model.predict(last_month_features)[0]

        avg_sales = df_to_use['Total_Sales'].mean()
        max_sales = df_to_use['Total_Sales'].max()

        result = {
            "type": result_type,
            "name": name_for_model,
            "sales_prediction": round(prediction, 2),
            "context": {
                "avg_sales": round(avg_sales, 2),
                "max_sales": round(max_sales, 2)
            }
        }
        if reason:
            result["reason"] = reason
            
        return json.dumps(result, ensure_ascii=False)

if __name__ == '__main__':
    reg_model = RegressionModel()
    
    if reg_model.monthly_data is not None:
        print("\n--- 테스트 예측 ---")
        test_product = "Bush Somerset Collection Bookcase"
        result_json = reg_model.predict(test_product)
        print(f"'{test_product}' 예측 결과:")
        print(json.dumps(json.loads(result_json), indent=2, ensure_ascii=False))

        print("\n--- 'Chairs' 카테고리 전체 예측 ---")
        test_product_chairs = "Global Troy Executive Leather Low-Back Tilter"
        result_json_chairs = reg_model.predict(test_product_chairs)
        print(f"'{test_product_chairs}' 예측 결과 (Chairs 카테고리로 대체될 수 있음):")
        print(json.dumps(json.loads(result_json_chairs), indent=2, ensure_ascii=False)) 