import implicit
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from flask import Flask
import json

app = Flask(__name__)


mid_to_idx = {}
idx_to_mid = {}
uid_to_idx = {}
idx_to_uid = {}
model = implicit.als.AlternatingLeastSquares(factors=50)
path=''




@app.route('/similar/<int:pid>')
def similar_items(pid):
    idx = mid_to_idx[pid]
    s = ""
    related = model.similar_items(idx)
    recs = []
    for id, sim in related:
        s += "http://cdn.shopdiscountdivas.com/account/api/getProductDetails?app_version=1.3.1&prod_id="+str(idx_to_mid[id])
        s += "\n"
        recs.append(idx_to_mid[id])
    ret = {"recs": recs}
    return json.dumps(ret)
    

@app.route('/')
@app.route('/index')
def start():
    print("start learning")
# def build_model():
    # customers = pd.read_csv(path+'raw_customers.csv')
    # customers = customers[['cid','email']]
    # customers = customers.dropna() 
    # order_products = pd.read_csv(path+'raw_orders_products.csv')
    # order_products = order_products[['product_id', 'order_id']]
    # orders = pd.read_csv(path+'raw_orders.csv')
    # orders = orders[['order_id', 'email']]
    # orders = orders.dropna() 

    # product_email = order_products.join(orders.set_index('order_id'), on='order_id')
    # product_email = product_email.dropna()
    # cid_product = product_email.join(customers.set_index('email'), on='email')
    # cid_product = cid_product.dropna()
    # cid_product = cid_product[['cid', 'product_id']]
    # cid_product = cid_product.rename(index=str, columns={"product_id": "pid"})
    # cid_product['rating']=1

    cid_product = pd.read_csv(path+'prepared_data.csv')
    # cp = cid_product.drop(cid_product.index[np.arange(800000)])
    cp = cid_product

    
    for (idx, mid) in enumerate(cp.pid.unique().tolist()):
        mid_to_idx[mid] = idx
        idx_to_mid[idx] = mid
    

    for (idx, uid) in enumerate(cp.cid.unique().astype('int64').tolist()):
        uid_to_idx[uid] = idx
        idx_to_uid[idx] = uid

    
    def map_ids(row, mapper):
        return mapper[row]
    I = cp.cid.apply(map_ids, args=[uid_to_idx]).as_matrix()
    J = cp.pid.apply(map_ids, args=[mid_to_idx]).as_matrix()
    V = np.ones(I.shape[0])
    cp_sp = sparse.coo_matrix((V, (J, I)), dtype=np.float64)
    cp_sp = cp_sp.tocsr()

    # initialize a model
    

    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(cp_sp)
    print("success")
    return "test"


