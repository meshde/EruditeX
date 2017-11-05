import numpy as np
from Models import SentEmbd
from Helpers import nn_utils
from Helpers import utils

def SentEmbd_gru_step(embedder,prev_hid_state,wVec):
	
	params = {param : param.get_value() for param in embedder.params}

	x__W_upd = np.dot(params[embedder.W_inp_upd_in],wVec)
	print(utils.get_var_name(x__W_upd,locals()),x__W_upd)

	h__U_upd = np.dot(params[embedder.U_inp_upd_hid],prev_hid_state)
	print(utils.get_var_name(h__U_upd,locals()),h__U_upd)

	sum1 = x__W_upd + h__U_upd 
	# print("Sum1",sum1)
	# print(sum1.shape)

	# print(params[embedder.b_inp_upd].shape)
	sum2 = sum1 + params[embedder.b_inp_upd].reshape(3,1)
	# print("Sum2",sum2)

	zt = nn_utils.sigmoid_np(sum2)
	print(utils.get_var_name(zt,locals()),zt)

	x__W_res = np.dot(params[embedder.W_inp_res_in],wVec)
	print(utils.get_var_name(x__W_res,locals()),x__W_res)

	h__U_res = np.dot(params[embedder.U_inp_res_hid],prev_hid_state)
	print(utils.get_var_name(h__U_res,locals()),h__U_res)

	rt = nn_utils.sigmoid_np(x__W_res + h__U_res + params[embedder.b_inp_res].reshape(3,1))
	print(utils.get_var_name(rt,locals()),rt)

	x__W_hid = np.dot(params[embedder.W_inp_hid_in],wVec)
	print(utils.get_var_name(x__W_hid,locals()),x__W_hid)

	h__U_hid = np.multiply(rt,np.dot(params[embedder.U_inp_hid_hid],prev_hid_state))
	print(utils.get_var_name(h__U_hid,locals()),h__U_hid)

	curr_hid_state_int = np.tanh(x__W_hid + h__U_hid + params[embedder.b_inp_hid].reshape(3,1
	print(utils.get_var_name(curr_hid_state_int,locals()),curr_hid_state_int)

	z__prev_h = np.multiply(zt,prev_hid_state)
	print(utils.get_var_name(z__prev_h,locals()),z__prev_h)

	z__hid = np.multiply((1-zt),curr_hid_state_int)
	print(utils.get_var_name(z__hid,locals()),z__hid)

	ht = z__prev_h + z__hid
	print("Hidden State",ht)
	return