// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>


/*  Function dealing with the case when E is empty

 When E is empty, beta_0 is not unique and is between [Low, High]. We set beta_0 to be either Low or High depend on j is in L or R so that E becomes non-empty.
*/
arma::vec Empty_E(arma::vec a, arma::mat B, arma::vec c, arma::vec theta_w, arma::uvec L, arma::uvec R, int j_in_L, double lam) {
	
	arma::vec beta_0_temp;
	double beta_0_update = 0.0;
	int index_move_to_E = 0;
	arma::vec output(2); 
	arma::vec beta_w = arma::trans(B) * theta_w / lam;

	if (j_in_L) {
		// Set beta_0 as High
		beta_0_temp = (- B.rows(R) * beta_w - c.elem(R)) / a.elem(R);
		beta_0_update = beta_0_temp.min();
		index_move_to_E = (int) beta_0_temp.index_min();   
	} else {
		// Set beta_0 as Low
		beta_0_temp = (- B.rows(L) * beta_w - c.elem(L)) / a.elem(L);
		beta_0_update = beta_0_temp.max();
		index_move_to_E = - (int) beta_0_temp.index_max() - 1;
	}
	output[0] = beta_0_update;
	output[1] = index_move_to_E;
	return output;
}


// Update or downdate the inverse matrix by changing only one point
arma::mat w_path_update(arma::mat K_inv, arma::vec a, arma::mat B, arma::uvec E, int E_size) {
       
	arma::uword k = E_size;
	arma::uword index_insert = E[k-1];
	arma::mat K_inv_new(k+1, k+1);
	arma::uvec E_prev = E.subvec(0, k-2);
	arma::mat B_E_prev = B.rows(E_prev);
	
	arma::vec v = (B.row(index_insert)).t();
	arma::vec u1(k, arma::fill::ones);
	u1 *= a(index_insert);
    u1.tail(k-1) = B_E_prev * v;
	arma::vec u2 = K_inv* u1;	
	double d = 1.0/(arma::dot(v, v) - dot(u1, u2));	
	arma::vec u3 = d * u2;
	arma::mat F11_inv = K_inv + d * u2 * u2.t();
	K_inv_new(k, k) = d;
	K_inv_new(arma::span(0, k-1), k) = -u3;
	K_inv_new(k, arma::span(0, k-1)) = -u3.t();
	K_inv_new(arma::span(0, k-1), arma::span(0, k-1)) = F11_inv;

	return K_inv_new;
}

arma::mat w_path_downdate(arma::mat K_inv, arma::mat B, arma::uvec E_prev, int E_prev_size, int index_remove) {

	arma::uword k = E_prev_size;
	arma::mat K_inv_new(k, k);
	
	if (index_remove < k-1) {
		arma::rowvec tmpv1 = K_inv.row(index_remove+1);
		K_inv.rows(index_remove+1, k-1) = K_inv.rows(index_remove+2, k);
		K_inv.row(k) = tmpv1;
		arma::vec tmpv2 = K_inv.col(index_remove+1);
		K_inv.cols(index_remove+1, k-1) = K_inv.cols(index_remove+2, k);
		K_inv.col(k) = tmpv2;
	}
	arma::mat F11_inv = K_inv.submat(0, 0, k-1, k-1);
	double d = K_inv(k, k);
	arma::vec u = - K_inv(arma::span(0, k-1), k) / d;
	K_inv_new = F11_inv - d * u * u.t();
	return K_inv_new;
}

arma::vec Compute_b(arma::vec a, arma::mat B, arma::uvec E, int E_size, int j, double T, arma::mat K_inv) {
	arma::vec y_et(E_size + 1, arma::fill::zeros);
	y_et(0) = -a(j) * T;
    y_et.tail(E_size) = -B.rows(E) * B.row(j).t() * T;
    arma::vec b = K_inv * y_et;
    return b;
}

arma::mat Compute_K_inv(arma::vec a, arma::mat B, arma::uvec E, int E_size) {
	arma::mat B_elb = B.rows(E);
    arma::mat K(E_size + 1, E_size + 1, arma::fill::zeros);
    K(arma::span(1, E_size), 0) = a.elem(E) ;
    K(0, arma::span(1, E_size)) = a.elem(E).t();
    K(1,1,arma::size(E_size,E_size)) = B_elb * B_elb.t();
    arma::mat K_inv = arma::inv(K);
    return K_inv;
}

//'Case-weight adjusted solution path for L2 regularized nonsmooth problem (quantile regression and svm)
//'
//' @description
//' Path-following algorithm to exactly solve 
//' 	(beta_{0,w}, beta_{w}) = argmin_{beta_0, beta} \sum_{i \neq j} f(g_i(beta_0, beta)) + w*f(g_{j}(beta_0, beta)) + lambda / 2 * \|beta\|_2^2
//' for 0 <= w <= 1, where g_i(beta_0, beta) = a_i beta_0 + b_i^T beta + c_i and f(r) = alpha_0 max(r, 0) + alpha_1 max(-r, 0)
//' with initial E/L/R, residual, and inverse matrix given.
arma::mat case_path_nonsmooth_with_init(arma::vec a, arma::mat B, arma::vec c, double lam, double alpha_0, double alpha_1, int j, 
	double beta_0_w0, arma::vec beta_w0, arma::vec theta_w0, arma::uvec E_init, arma::uvec L_init, arma::uvec R_init, 
	arma::vec r_init, arma::mat K_inv_init){
	const int n = B.n_rows;
	const int p = B.n_cols;

	const int N = 10000;
	
  	// Store breakpoints and solutions
	//arma::mat Beta(p, N+1);
	//Beta.col(0) = beta_w0;
	arma::vec Beta_0(N+1);
	Beta_0(0) = beta_0_w0;
	arma::vec W_vec(N+1);
	W_vec(0) = 1; 
	arma::mat Theta;
	Theta.insert_cols(Theta.n_cols, theta_w0);
	arma::mat R_mat;
	R_mat.insert_cols(R_mat.n_cols, r_init);
	double beta_0_w = beta_0_w0;
	//arma::vec beta_w = beta_w0;
	arma::vec theta_w = theta_w0;
	arma::vec r = r_init;

	// Define \tilde B, a_j, tilde_B_j
/*	arma::mat tilde_B(n, p+1);
	tilde_B.col(0) = a;
	tilde_B.cols(1, p) = B;
	const double a_j = a(j);
	arma::vec tilde_B_j = tilde_B.row(j).t();*/
	
	// Declare and initialize three elbow sets, and theta
	arma::uvec E = E_init;
	arma::uvec L = L_init;
	arma::uvec R = R_init;
	int E_size = E.size();
	int L_size = L.size();
	int R_size = R.size();

	arma::uvec index_insert(1);
	int index_j = -1;
	int j_in_L = -1;
	arma::vec theta_insert(1);

    // Initialize E/L/R and theta_E
/*	const double epsilon = 1e-6;
	for (unsigned i=0; i<n; i++){
		index_insert(0) = i;
		if (fabs(theta_w0[i]+alpha_0) < epsilon){
			R.insert_rows(R_size, index_insert);
			R_size = R_size + 1;
			if (i == (unsigned) j){
				j_in_L = 0;
			}
		} else if (fabs(theta_w0[i]-alpha_1) < epsilon){
			L.insert_rows(L_size, index_insert);
			L_size = L_size + 1;
			if (i == (unsigned) j){
				j_in_L = 1;
			}
		} else {
			E.insert_rows(E_size, index_insert);
			if (i == (unsigned) j) {
			  index_j = E_size;  // The index of j in set E
			}
			E_size = E_size + 1;
		}
	}*/

    // Declare variables 
	int m = 0;
	double w_m = 1.0;
	double w_m_next = 1.0;

	double T = 0.0;
/*	arma::vec d_0m(1);
	arma::vec d_m;
	arma::vec temp1;
	arma::vec temp2;
	arma::vec temp3(p+1);*/
	arma::vec b_m;
	arma::vec h_m(n);
	arma::mat K_inv;

	arma::vec w_1_alpha1_temp;
	double w_1_alpha1_max = 0.0;
	int w_1_alpha1_index = 0;
	arma::vec w_1_alpha0_temp;
	double w_1_alpha0_max = 0.0;
	int w_1_alpha0_index = 0;
	double w_1_max = 0.0;
	int w_1_index = 0;

	arma::vec w_2_L_temp;
	double w_2_L_max = 0.0;
	int w_2_L_index = 0;
	arma::vec w_2_R_temp;
	double w_2_R_max = 0.0;
	int w_2_R_index = 0;
	double w_2_max = 0.0;
	int w_2_index = 0;
	int w_2_L_is_max = 1;
	const double INF = 1e8;

	int index_in_elbow = 0;
	arma::vec E_empty_output(2);
	double beta_0_update = 0.0;

	// Check where case j is located
	const double epsilon = 1e-5;
	if (fabs(theta_w0[j]+alpha_0) < epsilon) {
		j_in_L = 0;		
	} else if (fabs(theta_w0[j]-alpha_1) < epsilon) {
		j_in_L = 1;
	} else {
		for (arma::uword E_index=0; E_index<E_size; E_index++) {
			if (E[E_index] == j) {
				index_j = E_index;
				break;
			}
		}
	}

	// Initialize K_inv
	if (E_size) {
		K_inv = K_inv_init;
	}

	// Manage the case when j is in E
	if (j_in_L == -1) {
		// Find the next breakpoint
		if (theta_w(j) > 0){
			w_m = theta_w(j) / alpha_1;
			j_in_L = 1;
		} else {
			w_m = theta_w(j) / (-alpha_0);
			j_in_L = 0;
		}

		// Update three sets and theta_E
		if (w_m > 0){
			index_insert(0) = j;
			if (j_in_L){
				L.insert_rows(L_size, index_insert);
				L_size = L_size + 1;
			} else {
				R.insert_rows(R_size, index_insert);
				R_size = R_size + 1;
			}
			if (E_size > 1){
				K_inv = w_path_downdate(K_inv, B, E, E_size, index_j);
            }
			E.shed_row(index_j);
			//theta_E.shed_row(index_j);
			E_size = E_size - 1;
			r(j) = 0;
		}
		m = m + 1;
		//Beta.col(m) = beta_w0;
		Beta_0(m) = beta_0_w0;
		Theta.insert_cols(Theta.n_cols, theta_w);
		R_mat.insert_cols(R_mat.n_cols, r);
		W_vec(m) = w_m; 
	}

	// Case when alpha_0 = 0 and j is in R (Algorithm terminates)
	if (alpha_0 == 0 && j_in_L == 0){
		m = m + 1;
		//Beta.col(m) = Beta.col(m-1);
		Beta_0(m) = Beta_0(m-1);
		w_m = 0;
		W_vec(m) = w_m;
		Theta.insert_cols(Theta.n_cols, theta_w);
		R_mat.insert_cols(R_mat.n_cols, r);
	}

	// Compute T (It will not change)
	if (j_in_L) {
		T = alpha_1;
	} else {
		T = -alpha_0;
	}

/*	if (E_size) {
		tilde_B_Bt_inverse = arma::solve(tilde_B.rows(E)*arma::trans(tilde_B.rows(E)), arma::eye(E_size, E_size));
	}*/

	while (w_m > 1e-6) {

		if (E_size > 0) {
			// Compute b = (b_0, b_1, ..., b_{|E|}) and h_m
			b_m = Compute_b(a, B, E, E_size, j, T, K_inv);
			h_m = a * b_m(0) + B * (arma::trans(B.rows(E)) * b_m.tail(E_size) + B.row(j).t() * T);
/*     		// Compute three slopes d_0m, d_m, h_m
     		temp1 = tilde_B_Bt_inverse*(tilde_B.rows(E)*tilde_B_j);
     		temp2 = tilde_B_Bt_inverse*a.elem(E);
			d_0m = (a_j - a.elem(E).t()*temp1) / (a.elem(E).t()*temp2) * T;
			d_m = - temp2 * d_0m - temp1 * T;
			temp3 = arma::trans(tilde_B.rows(E))*d_m + T*tilde_B_j;
			h_m = a*d_0m + tilde_B * temp3;*/

			// Compute candidate w_1m
			// w_1_index is the case in E that should be moved to L/R, and w_1_max is the candidate w_1m
			w_1_max = -INF;
			if (E_size > 0) {
				w_1_alpha1_max = -INF;
				w_1_alpha1_temp = (- theta_w.elem(E) + alpha_1) / b_m.tail(E_size) + w_m;
				for (unsigned int i=0; i<E_size; i++) {
					if ((w_1_alpha1_temp(i) < w_m) && (w_1_alpha1_temp(i) > w_1_alpha1_max)) {
						w_1_alpha1_max = w_1_alpha1_temp(i);
						w_1_alpha1_index = i;
					}
				}

				w_1_alpha0_max = -INF;
				w_1_alpha0_temp = (- theta_w.elem(E) - alpha_0) / b_m.tail(E_size) + w_m;
				for (unsigned int i=0; i<E_size; i++) {
					if ((w_1_alpha0_temp(i) < w_m) && (w_1_alpha0_temp(i) > w_1_alpha0_max)) {
						w_1_alpha0_max = w_1_alpha0_temp(i);
						w_1_alpha0_index = i;
					}
				}

				if (w_1_alpha0_max > w_1_alpha1_max) {
					w_1_max = w_1_alpha0_max;
					w_1_index = w_1_alpha0_index;
				} else {
					w_1_max = w_1_alpha1_max;
					w_1_index = w_1_alpha1_index;
				}
			}

			// Compute candidate w_2m
			// w_2_index is the case in L/R that should be moved to E, and w_2_L_is_max indicates this case is from L or R
			// w_2_max is the candidate w_2m
			w_2_max = -INF;
			w_2_L_max = -INF;
			w_2_R_max = -INF;
			if (L_size > 0) {
				w_2_L_temp = - lam * r.elem(L) / h_m.elem(L) + w_m;
				for (unsigned int i=0; i<L_size; i++) {
					if ((w_2_L_temp(i) < w_m) && (w_2_L_temp(i) > w_2_L_max)) {
						w_2_L_max = w_2_L_temp(i);
						w_2_L_index = i;
					}
				}
			}

			if (R_size > 0) {
				w_2_R_temp = - lam * r.elem(R) / h_m.elem(R) + w_m;
				for (unsigned int i=0; i<R_size; i++) {
					if ((w_2_R_temp(i) < w_m) && (w_2_R_temp(i) > w_2_R_max)) {
						w_2_R_max = w_2_R_temp(i);
						w_2_R_index = i;
					}
				}
			}

			if (w_2_L_max > w_2_R_max) {
				w_2_max = w_2_L_max;
				w_2_index = w_2_L_index;
				w_2_L_is_max = 1;
			} else {
				w_2_max = w_2_R_max;
				w_2_index = w_2_R_index;
				w_2_L_is_max = 0;
			}

			w_m_next = std::max(w_1_max, w_2_max);
			
			// Compute beta, r at the next breakpoint
			beta_0_w += b_m(0) / lam * (w_m_next - w_m);
			theta_w.elem(E) += b_m.tail(E_size) * (w_m_next - w_m);
			theta_w(j) = w_m_next * T;
			r += h_m / lam * (w_m_next - w_m);
			//beta_w = beta_w + temp3.tail(p) * (w_m_next - w_m) / lam;

			// Update three elbow sets and theta_E
			if (w_m_next == w_1_max) {
				// The case when E moves an element to either L or R
				index_insert(0) = E[w_1_index];				
				if (fabs(theta_w[E[w_1_index]] - alpha_1) < epsilon) {
					L.insert_rows(L_size, index_insert);
					L_size = L_size + 1;
				} else {
					R.insert_rows(R_size, index_insert);
					R_size = R_size + 1;
				}
				if (E_size > 1){
					K_inv = w_path_downdate(K_inv, B, E, E_size, w_1_index);
                }
				r(E[w_1_index]) = 0;
				E.shed_row(w_1_index);
				//theta_E.shed_row(w_1_index);
				E_size = E_size - 1;
			} else {
				// The case when L/R moves an element to E
				if (w_2_L_is_max) {
					index_insert(0) = L[w_2_index];
					theta_insert(0) = alpha_1;
					E.insert_rows(E_size, index_insert);
					theta_w[L[w_2_index]] = alpha_1;
					//theta_E.insert_rows(E_size, theta_insert);
					E_size = E_size + 1;  
					L.shed_row(w_2_index);
					L_size = L_size - 1;
				} else {
					index_insert(0) = R[w_2_index];
					theta_insert(0) = -alpha_0;
					E.insert_rows(E_size, index_insert);
					theta_w[R[w_2_index]] = -alpha_0;
					//theta_E.insert_rows(E_size, theta_insert);
					E_size = E_size + 1;  
					R.shed_row(w_2_index);
					R_size = R_size - 1;					
				}
				K_inv = w_path_update(K_inv, a, B, E, E_size);
			}
		} else {
			// The case when E is empty
		    E_empty_output = Empty_E(a, B, c, theta_w, L, R, j_in_L, lam);
		    beta_0_update = E_empty_output[0];
		    r = r + (beta_0_update - beta_0_w) * a;   // Update the residual
		    beta_0_w = beta_0_update;
		    index_in_elbow = E_empty_output[1];
		    // Move an element from L/R to E
		    if (index_in_elbow < 0) {
					index_insert(0) = L[-index_in_elbow-1];
					theta_insert(0) = alpha_1;
					E.insert_rows(E_size, index_insert);
					theta_w[L[-index_in_elbow-1]] = alpha_1;
					//theta_E.insert_rows(E_size, theta_insert);
					E_size = E_size + 1;  
					L.shed_row(-index_in_elbow-1);
					L_size = L_size - 1;		
				} else {
					index_insert(0) = R[index_in_elbow];
					theta_insert(0) = -alpha_0;
					E.insert_rows(E_size, index_insert);
					theta_w[R[index_in_elbow]] = -alpha_0;
					//theta_E.insert_rows(E_size, theta_insert);
					E_size = E_size + 1;  
					R.shed_row(index_in_elbow);
					R_size = R_size - 1;
		    }
			Beta_0(m) = beta_0_w;

			K_inv = Compute_K_inv(a, B, E, E_size);			
			continue;
		}	
		m = m + 1;
		//Beta.col(m) = beta_w;
		Beta_0(m) = beta_0_w;
		W_vec(m) = w_m_next; 
		w_m = w_m_next;
		Theta.insert_cols(Theta.n_cols, theta_w);
		R_mat.insert_cols(R_mat.n_cols, r);
	}

	// arma::mat Beta_output(p, m+1);
	// Beta_output = Beta.cols(0, m);

	arma::vec Beta_0_output(m+1);
	Beta_0_output = Beta_0.subvec(0, m);

	arma::vec W_vec_output(m+1);
	W_vec_output = W_vec.subvec(0, m);

  	arma::mat output = join_cols(W_vec_output.t(), Beta_0_output.t(), R_mat, Theta);
	return output; 
}

// Record the time for w-path (Quantile regression)
// [[Rcpp::export(w_path_simulation)]]
double w_path_simulation(arma::vec a, arma::mat B, arma::vec c, double alpha_0, double alpha_1, arma::vec lam_list, arma::vec beta_0_w0_list, arma::mat beta_w0_mat, arma::mat theta_w0_mat) {
    
	const arma::uword n = B.n_rows;
	const arma::uword p = B.n_cols;
	arma::uword lam_length= lam_list.n_elem;
	double lam = 0.0;
	double beta_0_w0 = 0.0;
	arma::vec beta_w0; beta_w0.zeros(p);
	arma::vec theta_w0; theta_w0.zeros(n);

	arma::mat w_path_output;
	double grid_prop = 0.0;
	arma::uword out_ncol = 0;
	//double beta_0_LOO = 0.0;
	//arma::vec beta_LOO; beta_LOO.zeros(p);

	arma::vec Lam_rcv(lam_length, arma::fill::zeros);
	//arma::mat predict; predict.zeros(n, lam_length);
	arma::mat residual; residual.zeros(n, lam_length);

	clock_t begin, end;
	clock_t begin_init, end_init;
	double total_time = 0.0;
	double init_time = 0.0;

	arma::mat tilde_B(n, p+1);
	tilde_B.col(0) = a;
	tilde_B.cols(1, p) = B;
	arma::uvec index_insert(1);
	arma::vec theta_insert(1);
	arma::vec r_init(n);
	int E_size = 0;
	int L_size = 0;
	int R_size = 0;
	arma::mat K_inv_init;

	arma::vec zeros(n, arma::fill::zeros);

	//std::cout << "Check point 1" << "\n";

	begin = clock();
	for (arma::uword ind_lam = 0; ind_lam < lam_length; ind_lam++) {
		lam = lam_list[ind_lam];
		beta_0_w0 = beta_0_w0_list[ind_lam];
		beta_w0 = beta_w0_mat.col(ind_lam);
		theta_w0 = theta_w0_mat.col(ind_lam);

		// Initialize E/L/R, residual and matrix inversion
		arma::uvec E_init;
		arma::uvec L_init;
		arma::uvec R_init;
		E_size = 0;
		L_size = 0;
		R_size = 0;
		//arma::mat B_elb;

		begin_init = clock();
		//std::cout << "Check point 2" << "\n";
		const double epsilon = 1e-5;
		for (arma::uword i=0; i<n; i++){
			index_insert(0) = i;
			if (fabs(theta_w0[i]+alpha_0) < epsilon){
				R_init.insert_rows(R_size, index_insert);
				R_size = R_size + 1;
			} else if (fabs(theta_w0[i]-alpha_1) < epsilon){
				L_init.insert_rows(L_size, index_insert);
				L_size = L_size + 1;
			} else {
				E_init.insert_rows(E_size, index_insert);
				E_size = E_size + 1;
			}
		}
		//std::cout << "Check point 3" << "\n";
		r_init = beta_0_w0 * a + B * beta_w0 + c;
		//std::cout << "Check point 4" << "\n";

		if (E_size) {
			//std::cout << "Check point 5" << "\n";
			K_inv_init = Compute_K_inv(a, B, E_init, E_size);	
		} else {
			//std::cout << "Check point 6" << "\n";
			// Assign tilde_B_Bt_inverse_init with a random matrix (We choose identity matrix)
			K_inv_init = arma::eye(1, 1);
		}
		end_init = clock();
		init_time += double(end_init - begin_init)/CLOCKS_PER_SEC;

		for (arma::uword j = 0; j < n; j++) {
			//std::cout << "Check point with j =  " << j << " lam_index = " << ind_lam<< "\n";
			w_path_output = case_path_nonsmooth_with_init(a, B, c, lam, alpha_0, alpha_1, (int) j, beta_0_w0, beta_w0, theta_w0, 
				E_init, L_init, R_init, r_init, K_inv_init);
/*			if ((ind_lam % 10 < 3) && (j % (n/3) == 0)) {
				std::ostringstream file;
				file <<"w_path_check_n"<< n << "_p" << p << "_lamindex"<< ind_lam << "_case" << j<<".txt"; 
				std::string filename = file.str();
				w_path_output.save(filename, arma::raw_ascii);
			}*/
			out_ncol = w_path_output.n_cols;
			grid_prop = (0 - w_path_output(0, out_ncol-1)) / (w_path_output(0, out_ncol-2) - w_path_output(0, out_ncol-1));
			/*beta_0_LOO = grid_prop*w_path_output(1, out_ncol-2) + (1 - grid_prop)*w_path_output(1, out_ncol-1);
			beta_LOO = grid_prop*w_path_output(arma::span(2, p+1), out_ncol-2) + (1 - grid_prop)*w_path_output(arma::span(2, p+1), out_ncol-1);
			predict(j, ind_lam) = arma::dot(-B.row(j), beta_LOO.t()) + beta_0_LOO;*/
			residual(j, ind_lam) = grid_prop*w_path_output(j+2, out_ncol-2) + (1 - grid_prop)*w_path_output(j+2, out_ncol-1);

		} // end cases for loop

	} // end lam_list for loop
	end = clock();
	total_time = double(end - begin)/CLOCKS_PER_SEC;

	for (arma::uword k = 0; k < lam_length; k++) {
		//Lam_rcv[k] = arma::sum(arma::square(c - predict.col(k)))/n;
		
		//Lam_rcv[k] = arma::sum(arma::square(residual.col(k)))/n;

		// For quantile regression, alpha_0 = tau, alpha_1 = 1 - tau
		Lam_rcv[k] = arma::sum(alpha_0 * arma::max(residual.col(k), zeros) + alpha_1 * arma::max(-residual.col(k), zeros) )/n;
	}
	std::cout<< "lam_opt is "<< lam_list(Lam_rcv.index_min()) << "\n";
	std::cout<< "lam_opt_mse is "<< Lam_rcv.min() << "\n";
	std::cout<< "Initial time is "<< init_time << "\n";
	return total_time;
}

