import os, shutil, tempfile
import networkx as nx
import numpy as np
from scipy.stats import expon
import networkx as nx
import matplotlib.pyplot as plt
from numpy import loadtxt
from copy import deepcopy
from numpy import zeros, ix_, outer
from scipy.stats import wishart
from numpy.linalg import LinAlgError
from copy import deepcopy
from scipy.stats import geom
from math import log
from copy import deepcopy
from numpy import insert, delete, ix_, dtype, loadtxt
import os
import hashlib
import pickle
from numpy import array, mean, zeros, diag, savetxt, nan, isnan, nanmean, identity, diag, log, outer, square
from numpy import sum as npsum
import warnings
from numpy.random import choice
from numpy.linalg import norm as npnorm
from numpy import var as npvar
from scipy.optimize import minimize_scalar
from pathos.multiprocessing import Pool


def variance_mean_based(sample_of_matrices):
    mean_wishart=  mean(sample_of_matrices, axis=0)
    var_wishart= npvar(sample_of_matrices, axis=0)
    var_rom_mean_wishart=square(mean_wishart)+outer(diag(mean_wishart),diag(mean_wishart))

    # print(mean_wishart, var_wishart, var_rom_mean_wishart)

    def fff(exx):
        return(log(npnorm(var_wishart-var_rom_mean_wishart/exx)**2))

    yyy = minimize_scalar(fff, method='bounded', bounds=[0,1000000]).x
    # print(yyy)
    assert yyy > 2.0, "Bootstrap Optimization Failed"
    assert yyy < 999900, "Bootstrap Optimization Failed"
    return yyy
    #maximizes the function, function in the interval [lower_limit,\infty).

def get_partitions(lines, blocksize):
    list_of_lists=[]
    for i in range(0,len(lines)-blocksize, blocksize):
        list_of_lists.append(lines[i:(i+blocksize)])
    if len(lines)%blocksize == 0:
        list_of_lists.append(lines[-blocksize:])
    return list_of_lists

def combine_covs(tuple_covs, indices):
    cov_sum, scale_sum=0.0,0.0
    for i in indices:
        cov_sum+=tuple_covs[i][0]
        scale_sum+=tuple_covs[i][1]
    return cov_sum/scale_sum

def make_covariances(filenames, varcovfilename, cores, **kwargs):
    covs=[]
    p=Pool(cores)
    def t(filename):
        return empirical_covariance_wrapper_directly(filename, varcovfilename, **kwargs)
    covs=list(map(t, filenames))
    try:
        for fil in filenames:
            os.remove(fil)
    except OSError as e:
        warnings.warn('Erasing the files did not succeed',UserWarning)
    return covs

def make_single_files(filename,blocksize, verbose_level='normal', fileprefix = ""):
    filenames=[]
    os.mkdir(fileprefix + os.sep + "temp_adbayes")
    filename_reduced=fileprefix + os.sep  + "temp_adbayes" + os.sep  + filename.split(os.sep)[-1]+'boot.'
    with open(filename, 'r') as f:
        first_line=f.readline()
        lines=f.readlines()
    n=len(lines)
    line_sets=get_partitions(lines, blocksize)
    if verbose_level!='silent':
        print('total number of SNPs: '+ str(n))
    for i,lins in enumerate(line_sets):
        new_filename= filename_reduced+str(i)
        with open(new_filename, 'w') as g:
            g.write(first_line)
            g.writelines(lins)
        gzipped_filename=new_filename
        filenames.append(gzipped_filename)
    return filenames

def estimate_degrees_of_freedom_scaled_fast(filename, varcovfilename = "", bootstrap_blocksize=1000, ########################## 1000
                                            cores=1,  verbose_level='normal',  **kwargs):

    single_files=make_single_files(filename, blocksize=bootstrap_blocksize, verbose_level=verbose_level, fileprefix = (varcovfilename[0:(len(varcovfilename)-24)]))
    assert len(single_files)>1, 'There are ' +str(len(single_files)) + ' bootstrapped SNP blocks and that is not enough. Either add more data or lower the --bootstrap_blocksize'
    if len(single_files)<39:
        warnings.warn('There are only '+str(len(single_files))+' bootstrap blocks. Consider lowering the --bootstrap_blocksize or add more data.', UserWarning)
    single_covs=make_covariances(single_files, varcovfilename=varcovfilename, cores=cores, return_also_mscale=True, **kwargs)

    # print("single covs", hash_object(single_covs))

    # p=Pool(cores)
    def t(empty):
        indices=choice(len(single_covs), size=len(single_covs), replace = True)
        # indices=choice(len(single_covs), size=len(single_covs), replace = False)
        return combine_covs(single_covs, indices)

    # print(hash_object(list(map(t, list(range(100))))))
    return variance_mean_based(list(map(t, list(range(100)))))

def reduce_covariance(covmat):
    reducer=insert(identity(covmat.shape[0]-1), 0, -1, axis=1)
    return reducer.dot(covmat).dot(reducer.T)

def nan_divide(dividend, divisor):
    res=zeros(dividend.shape)
    for i in range(dividend.shape[0]):
        for j in range(dividend.shape[1]):
            if divisor[i,j]==0:
                res[i,j]=nan
            else:
                res[i,j]=dividend[i,j]/divisor[i,j]
    return res

def nan_inner_product(a,b):
    N=len(a)
    nans=0
    res=0
    for ai,bi in zip(a,b):
        if isnan(ai) or isnan(bi):
            nans+=1
        else:
            res+=ai*bi
    if N==nans:
        warnings.warn('There is an entry in the covariance matrix that is set to 0 because all the relevant data was nan.', UserWarning)
        return 0

    return res/(N-nans)

def nan_product(A,B):
    res=zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            res[i,j]=nan_inner_product(A[i,:], B[:,j])
    return res

def m_scaler(allele_freqs):
    s=nanmean(allele_freqs, axis=0)
    scaler=nanmean(s*(1.0-s))
    return scaler

def var(p,n):
    entries=array([pi*(1-pi)/(ni-1) for pi,ni in zip(p,n) if ni>1])
    return mean(entries)

def reduced_covariance_bias_correction(p,n,n_outgroup=0):
    Bs=[]
    for pi,ni in zip(p,n):
        Bs.append(var(pi,ni))
    outgroup_b=Bs.pop(n_outgroup)
    return diag(array(Bs))+outgroup_b

class ScaledEstimator(object):
    def __init__(self, add_variance_correction_to_graph=False, save_variance_correction=True, nodes=None, varcovname = ""):
        self.add_variance_correction_to_graph=add_variance_correction_to_graph
        self.nodes=nodes
        self.save_variance_correction=save_variance_correction
        self.variancecorrectionname = varcovname

    def subtract_ancestral_and_get_outgroup(self,p):
        return p-p[0,:]

    def __call__(self, xs, ns, extra_info={}):
        if 0 in ns:
            warnings.warn('There were 0s in the allele-totals, inducing nans and slower estimation.', UserWarning)
            ps=nan_divide(xs, ns)
        else:
            ps=xs/ns
        return self.estimate_from_p(ps, ns=ns, extra_info=extra_info)

    def estimate_from_p(self, p, ns=None, extra_info={}):
        p2 = self.subtract_ancestral_and_get_outgroup(p)
        if npsum(isnan(p2))>0:
            warnings.warn('Nans found in the allele frequency differences matrix => slower execution', UserWarning)
            m=nan_product(p2, p2.T)
        else:
            m=p2.dot(p2.T)/p2.shape[1]

        scaling_factor=m_scaler(p)
        extra_info['m_scale']=scaling_factor
        m=m/scaling_factor
        m=reduce_covariance(m)
        b=reduced_covariance_bias_correction(p, ns, 0)/scaling_factor
        if self.save_variance_correction:
            savetxt(self.variancecorrectionname, b)
        return m

def read_freqs(new_filename):
    with open(new_filename, 'r') as f:
        names=f.readline().split()
        pop_sizes=[]
        minors=[]
        for n,r in enumerate(f.readlines()):
            minor_majors=r.split()
            minor_list=[]
            pop_sizes_SNP=[]
            for minor_major in minor_majors:
                minor, major= list(map(float,minor_major.split(',')))
                minor_list.append(minor)
                pop_sizes_SNP.append(major+minor)
            minors.append(minor_list)
            pop_sizes.append(pop_sizes_SNP)
    return names, pop_sizes, minors

def get_xs_and_ns_from_treemix_file(snp_file):
    names, ns, minors= read_freqs(snp_file)
    xs=array(minors, dtype=dtype(float)).T
    ns=array(ns, dtype=dtype(float)).T
    return xs,ns,names

def order_covariance(xnn_tuple, outgroup=''):
    xs,ns,names=xnn_tuple
    assert outgroup in names, 'The outgroup was not found in the data. Did you spell it correctly?'
    n_outgroup=next((n for n, e in enumerate(names) if e==outgroup))
    xs_o=xs[n_outgroup,:]
    ns_o=ns[n_outgroup,:]
    names_o=names[n_outgroup]
    xs=delete(xs, n_outgroup,0)
    ns=delete(ns, n_outgroup,0)
    names.remove(names_o)
    xs=insert(xs, 0, xs_o, axis=0)
    ns=insert(ns, 0, ns_o, axis=0)
    names=[names_o]+names
    return xs,ns,names

def _get_permutation(actual, target):
    val_to_index={val:key for key, val in enumerate(actual)}
    indices_to_get_target=[val_to_index[val] for val in target]
    return indices_to_get_target

def reorder_reduced_covariance(cov, names, full_nodes, outgroup=''):
    names2=deepcopy(names)
    full_nodes2=deepcopy(full_nodes)
    names2.remove(outgroup)
    full_nodes2.remove(outgroup)
    indices=_get_permutation(names2, full_nodes2)
    return cov[ix_(indices, indices)]

def emp_cov_to_file(m, filename='emp_covimport', nodes=None):
    with open(filename, 'w') as f:
        f.write(' '.join(nodes)+'\n')
        for i, node in enumerate(nodes):
            f.write(node+ ' '+ ' '.join(map(str, m[i]))+'\n')

def make_estimator( nodes,  reducer, add_variance_correction_to_graph=False, save_variance_correction=True, varcovname = ""):
    return ScaledEstimator(add_variance_correction_to_graph=add_variance_correction_to_graph, save_variance_correction=save_variance_correction, varcovname=varcovname)

def rescale_empirical_covariance(m):
    n=m.shape[0]
    max_expected_trace=n*(n+1)/2-1
    multiplier= max_expected_trace/m.trace()

    return m*multiplier, multiplier

def empirical_covariance_wrapper_directly(snp_data_file, varcovfilename, **kwargs):
    xnn_tuple=get_xs_and_ns_from_treemix_file(snp_data_file)
    return xnn_to_covariance_wrapper_directly(xnn_tuple, varcovfilename, **kwargs)

def xnn_to_covariance_wrapper_directly(xnn_tuple, varcovfilename, **kwargs):
    est_args=kwargs['est']
    xnn_tuple=order_covariance(xnn_tuple, outgroup=est_args['reducer'])
    xs,ns,names=xnn_tuple
    est_args['varcovname']  = varcovfilename

    est= make_estimator( **est_args)
    extra_info_dic={}
    cov=est(xs,ns, extra_info_dic)
    cov=reorder_reduced_covariance(cov, names, est_args['nodes'], outgroup=est_args['reducer'])
    if ('add_variance_correction_to_graph' in est_args and
        est_args['add_variance_correction_to_graph'] and
        'save_variance_correction' in est_args and
        est_args['save_variance_correction']):
        filename=varcovfilename
        vc=loadtxt(filename)
        vc=reorder_reduced_covariance(vc, names, est_args['nodes'], outgroup=est_args['reducer'])
        savetxt(filename, vc)
    if 'return_also_mscale' in kwargs and kwargs['return_also_mscale']:
        return cov, extra_info_dic['m_scale']
    return cov

def get_covariance(input, varcovfilename= "", full_nodes=None, reduce_covariance_node=None, estimator_arguments={}, filename = ""):
    kwargs={}
    after_reduce_nodes=deepcopy(full_nodes)
    after_reduce_nodes.remove(reduce_covariance_node)
    kwargs['est']=estimator_arguments

    statistic = input
    print('INPUT: ', input)
    statistic = empirical_covariance_wrapper_directly(statistic, varcovfilename, **kwargs)

    statistic=rescale_empirical_covariance(statistic)
    emp_cov_to_file(statistic[0], filename, after_reduce_nodes)
    with open(filename, 'a') as f:
        f.write('multiplier='+str(statistic[1]))
    return statistic

def rename_key(tree, old_key_name, new_key_name):
    node=tree[old_key_name]
    tree[new_key_name]=node
    ps= get_real_parents(node)
    for p in ps:
        if not is_root(p):
            tree[p]=_rename_child(tree[p], old_key_name, new_key_name)
    cs=get_real_children(node)
    for c in cs:
        tree[c]=rename_parent(tree[c], old_key_name, new_key_name)
    del tree[old_key_name]
    return tree

def get_number_of_ghost_populations(tree):
    count=0
    for key,node in list(tree.items()):
        if node_is_coalescence(node):
            child1,child2=get_children(node)
            if node_is_admixture(tree[child1]) and node_is_admixture(tree[child2]):
                count+=1
    return count

def update_all_branches(tree, updater):
    for key, node in list(tree.items()):
        if node_is_admixture(node):
            node[3]+=updater()
            node[4]+=updater()
            if node[3] < 0:
                node[3] = -node[3]
            if node[4] < 0:
                node[4] = -node[4]
        else:
            node[3]+=updater()
            if node[3] < 0:
                node[3] = -node[3]
    return tree

def update_branch_length(tree,key,branch, new_length):
    tree[key][branch+3]=new_length

def remove_root_attachment(tree, orphanota_key, orphanota_branch):
    '''
    The situation is different when the root is removed because of the special naming strategy.

                r
               / \
             /    \
      orphanota   new_root
    Here a new root is born.

    '''
    rooted_keys=find_rooted_nodes(tree)
    for key,branch,len_to_root in rooted_keys:
        if key!=orphanota_key or orphanota_branch!=branch:
            if node_is_coalescence(tree[key]):
                tree=rename_root(tree, key)
                r=len_to_root
                del tree[key]
            else:
                tree[key],r=get_branch_length_and_reset(tree[key], 'r', 'closed_branch')
            tree[orphanota_key][orphanota_branch]=None
    return tree,'r', r,None

def get_branch_length_and_reset(node, parent_key, new_length,add=False):
    if node[0]==parent_key:
        old_length=node[3]
        if add:
            node[3]+=new_length
        else:
            node[3]=new_length
        return node, old_length
    elif node[1]==parent_key:
        old_length=node[4]
        if add:
            node[4]+=new_length
        else:
            node[4]=new_length
        return node, old_length
    else:
        assert False, 'could not give new length because the parent was unknown'

def rename_root(tree, old_name):
    for _,node in list(tree.items()):
        if node[0]==old_name:
            node[0]='r'
        if (node[1] is not None and node[1]==old_name):
            node[1]='r'
    return tree

def _rename_child(node, old_name, new_name):
    if node[5]==old_name:
        node[5]=new_name
    elif node[6]==old_name:
        node[6]=new_name
    else:
        assert False, "tried to rename a child that didn't exist in its parents documents."
    return node

def rename_parent(node, old_name, new_name):
    if node[0]==old_name:
        node[0]=new_name
    elif node[1]==old_name:
        node[1]=new_name
    return node

def find_rooted_nodes(tree):
    res=[]
    for key,node in list(tree.items()):
        if node[0]=='r' or (node[1] is not None and node[1]=='r'):
            if node[0]=='r':
                res.append((key,0,node[3]))
            else:
                res.append((key,1,node[4]))
    return res

def _find_rooted_branches(tree):
    res=[]
    for key,node in list(tree.items()):
        if node[0]=='r':
            res.append((key,0))
        if node[1] is not None and node[1]=='r':
            res.append((key,1))
    return res

def _get_root_sibling(tree, child_key, child_branch):
    root_keys=_find_rooted_branches(tree)
    if len(root_keys)==1:
        return child_key, child_branch
    else:
        if root_keys[0][0]==child_key and root_keys[0][1]==child_branch:
            return root_keys[1]
        else:
            return root_keys[0]

def node_is_non_admixture(node):
    return (node[1] is None)

def node_is_admixture(node):
    return (node[1] is not None)

def node_is_coalescence(node):
    return (node[1] is None and node[5] is not None)

def node_is_leaf_node(node):
    return (node[1] is None and node[5] is None)

def get_all_branches(tree):
    res=[]
    for key, node in list(tree.items()):
        if node_is_admixture(node):
            res.extend([(key, 0),(key,1)])
        else:
            res.append((key,0))
    return res

def update_specific_branch_lengths(tree, branches, new_lengths, add=False):
    if add:
        for (key, branch), new_length in zip(branches,new_lengths):
            tree[key][branch+3]+=new_length
            if tree[key][branch+3]<0:
                return None
    else:
        for (key, branch), new_length in zip(branches,new_lengths):
            tree[key][branch+3]=new_length
            if tree[key][branch+3]<0:
                return None
    return tree

def get_all_branch_descendants_and_rest(tree, key,branch):
    all_branches=get_all_branches(tree)
    descendant_branches=_get_descendant_branches(tree, key,branch)
    return list(set(all_branches)-set(descendant_branches))

def _get_descendant_branches(tree, key, branch):
    child_key1=tree[key][5]
    if child_key1 is None: #branch is leaf
        return [(key,0)]
    else:
        child_key1_branch=mother_or_father(tree, child_key1, key)
        ans=[(key, branch)]+_get_descendant_branches(tree, child_key1, child_key1_branch)
        child_key2=tree[key][6]
        if child_key2 is None:
            return ans
        return ans+_get_descendant_branches(tree, child_key2, mother_or_father(tree,child_key2, key))

def get_other_children(node, child_key):
    res=[]
    for n in get_children(node):
        if n is not None and n!=child_key:
            res.append(n)
    return res

def get_children(node):
    return node[5:7]

def _get_index_of_parent(node, parent):
    if node[0]==parent:
        return 0
    if node[1]==parent:
        return 1
    return -1

def screen_and_prune_one_in_one_out(tree):
    keys_to_check=list(tree.keys())
    for key in keys_to_check:
        if key in tree:
            node=tree[key]
            children=get_real_children(node)
            parents=get_real_parents(node)
            if len(children)==1 and len(parents)==1:
                tree=remove_one_in_one_out(tree, children[0], key, parents[0])
    return tree

def remove_one_in_one_out(tree, child, key, parent_key):
    branch=mother_or_father(tree, key, parent_key)
    length=get_branch_length(tree, key, branch)
    child_branch=mother_or_father(tree, child, key)
    child_length=get_branch_length(tree, child, child_branch)
    tree=update_parent_and_branch_length(tree, child, child_branch, new_parent=parent_key, new_branch_length=child_length+length)
    if parent_key!='r':
        tree[parent_key]=_update_child(tree[parent_key], key, child)
    del tree[key]
    return tree

def get_leaf_keys(tree):
    res=[]
    for key, node in list(tree.items()):
        if node_is_leaf_node(node):
            res.append(key)
    return res

def get_categories(tree):
    leaves=[]
    admixture_nodes=[]
    for key, node in list(tree.items()):
        if node_is_leaf_node(node):
            leaves.append(key)
        if node_is_admixture(node):
            admixture_nodes.append(key)
    return leaves, admixture_nodes

def get_parent_of_branch(tree, key, branch):
    return tree[key][branch]

def get_branch_length(tree,key,branch):
    return tree[key][branch+3]

def get_admixture_proportion_from_key(tree, key):
    return tree[key][2]

def get_admixture_proportion(tree, child_key,child_branch):
    key=get_parent_of_branch(tree, child_key,child_branch)
    return tree[key][2]

def update_parent_and_branch_length(tree, child_key, child_branch, new_parent, new_branch_length):
    tree[child_key][child_branch]=new_parent
    tree[child_key][child_branch+3]=new_branch_length
    return tree

def get_admixture_keys_and_proportions(tree):
    keys=[]
    props=[]
    for key, node in list(tree.items()):
        if node_is_admixture(node):
            keys.append(key)
            props.append(node[2])
    return keys, props

def get_destination_of_lineages(tree, ready_lineages):
    single_coalescences={} #list of tuples ((key,branch),(sister_key,sister_branch))
    double_coalescences=[]
    admixtures=[]
    for key, branch in ready_lineages:
        if (key,branch) in single_coalescences:
            double_coalescences.append(((key,branch),single_coalescences[(key,branch)]))
            del single_coalescences[(key,branch)]
            continue
        parent_key=tree[key][branch]
        if parent_key=='r':
            sister_key, sister_branch=_get_root_sibling(tree, key, branch)
            single_coalescences[(sister_key,sister_branch)]=(key,branch)
            continue
        parent=tree[parent_key]
        if node_is_coalescence(parent):
            sister_key, sister_branch=get_sister_branch(tree, parent,key, branch)
            single_coalescences[(sister_key,sister_branch)]=(key,branch)
        elif node_is_admixture(parent):
            admixtures.append((key,branch))
        else:
            assert False, 'the parent of a node was neither admixture nor coalescence'
    return double_coalescences, single_coalescences, admixtures

def get_sister_branch(tree, parent, key, branch):
    if parent[5]==parent[6]:
        return key, other_branch(branch)
    else:
        if parent[5]==key:
            return parent[6], mother_or_father(tree, parent[6], tree[key][branch])
        elif parent[6]==key:
            return parent[5], mother_or_father(tree, parent[5], tree[key][branch])

def propagate_married(tree, list_of_pairs):
    res=[]
    for (key1,branch1),(_,_) in list_of_pairs:
        parent_key=get_parent_of_branch(tree, key1, branch1)
        res.append((parent_key,0))
    return res

def propagate_admixtures(tree, list_of_admixtures):
    res=[]
    for key,branch in list_of_admixtures:
        parent_key=get_parent_of_branch(tree, key, branch)
        res.append((parent_key,0))
        res.append((parent_key,1))
    return res

def mother_or_father(tree, child_key, parent_key):
    if tree[child_key][0]==parent_key:
        return 0
    elif tree[child_key][1]==parent_key:
        return 1

def insert_children_in_tree(tree):
    children={key:[] for key in tree}
    for key in tree:
        parents = get_real_parents(tree[key])
        for parent in parents:
            if parent!='r':
                children[parent].append(key)
    for key in tree:
        tree[key]=_update_parents(tree[key], children[key])
    return tree

def get_all_branch_lengths(tree):
    res=[]
    for key, node in list(tree.items()):
        if node_is_non_admixture(node):
            res.append(node[3])
        else:
            res.extend(node[3:5])
    return res

def get_all_admixture_proportions(tree):
    res=[]
    for key, node in list(tree.items()):
        if node_is_admixture(node):
            res.append(node[2])
    return res

def graft(tree, remove_key, add_to_branch, insertion_spot, new_node_code, which_branch, remove_branch=0):
    #if we are at the root, things are different.
    if add_to_branch=='r':
        return graft_onto_root(tree, insertion_spot, remove_key, new_name_for_old_root=new_node_code, remove_branch=remove_branch)

    #updating the grafted_branch. Easy.
    tree[remove_key][remove_branch]=new_node_code

    #dealing with the other child of the new node
    index_of_pop_name_to_change=which_branch
    index_of_branchl_to_change=which_branch+3
    #saving info:
    parent_of_branch=tree[add_to_branch][index_of_pop_name_to_change]
    length_of_piece_to_break_up=tree[add_to_branch][index_of_branchl_to_change]
    #updating node
    tree[add_to_branch][index_of_pop_name_to_change]=new_node_code
    tree[add_to_branch][index_of_branchl_to_change]=length_of_piece_to_break_up*insertion_spot

    #taking care of the new node
    tree[new_node_code]=[parent_of_branch, None, None, length_of_piece_to_break_up*(1.0-insertion_spot), None, add_to_branch, remove_key]

    #taking care of the parent of the new node
    if parent_of_branch != 'r':
        tree[parent_of_branch]=_update_child(tree[parent_of_branch], add_to_branch, new_node_code)

    return tree

def graft_onto_root(tree, insertion_spot, remove_key, new_name_for_old_root, remove_branch=0):
    root_keys=find_rooted_nodes(tree)

    if len(root_keys)==1:#this is the special case where an admixture leads to the new root
        return graft_onto_rooted_admixture(tree, insertion_spot, remove_key, root_keys[0], remove_branch=remove_branch)

    #updating the grafted_branch. Easy.
    tree[remove_key][remove_branch]='r'

    #dealing with the other child of the new node, but since the new node is the root, the old root is the new node. If that makes sense.
    tree[new_name_for_old_root]=['r', None, None, insertion_spot, None,root_keys[0][0], root_keys[1][0]]

    #dealing with the children of the new node.
    for rkey, _, b_l in root_keys:
        tree[rkey]=_update_parent(tree[rkey], 'r', new_name_for_old_root)

    return tree

def materialize_root(tree, new_key):
    (child_key1, child_branch1,_),(child_key2, child_branch2,_) = find_rooted_nodes(tree)
    tree[new_key]=['r',None,None,None,None,child_key1, child_key2]
    tree[child_key1][child_branch1]=new_key
    tree[child_key2][child_branch2]=new_key
    return tree

def move_node(tree, regraft_key, regraft_branch, parent_key, distance_to_regraft, chosen_piece, new_node_name='x'):
    if parent_key=='r':
        sister_key,sister_branch= _get_root_sibling(tree, regraft_key, regraft_branch)
        if chosen_piece.child_key=='r':
            u1,_=chosen_piece.get_leaf_and_root_sided_length(distance_to_regraft)
            tree[sister_key][sister_branch+3]=tree[sister_key][sister_branch+3]+u1
        elif chosen_piece.child_key==sister_key and chosen_piece.child_branch==sister_branch:
            u1,_=chosen_piece.get_leaf_and_root_sided_length(distance_to_regraft)
            tree[sister_key][sister_branch+3]=u1
        else:
            tree=rename_root(tree, sister_key)
            del tree[sister_key]
            u1,u2 =chosen_piece.get_leaf_and_root_sided_length(distance_to_regraft)
            if chosen_piece.parent_key == sister_key:
                tree[new_node_name]=['r', None,None,u2,None,regraft_key,chosen_piece.child_key]
            else:
                tree[new_node_name]=[chosen_piece.parent_key, None,None,u2,None,regraft_key,chosen_piece.child_key]
            if chosen_piece.parent_key=='r':
                pass
            elif chosen_piece.parent_key != sister_key:
                tree[chosen_piece.parent_key]=_rename_child(tree[chosen_piece.parent_key], chosen_piece.child_key, new_node_name)
            tree=update_parent_and_branch_length(tree, chosen_piece.child_key, chosen_piece.child_branch, new_node_name, u1)
            tree[regraft_key][regraft_branch]=new_node_name
    else:
        if chosen_piece.child_key=='r':
            tree=materialize_root(tree, new_node_name)
            u1,_=chosen_piece.get_leaf_and_root_sided_length(distance_to_regraft)
            tree[new_node_name][3]=u1
            tree[regraft_key][regraft_branch]='r'
        else:
            u1,u2=chosen_piece.get_leaf_and_root_sided_length(distance_to_regraft)
            tree[new_node_name]=[chosen_piece.parent_key, None,None,u2,None,regraft_key,chosen_piece.child_key]
            if chosen_piece.parent_key=='r':
                pass
            else:
                tree[chosen_piece.parent_key]=_rename_child(tree[chosen_piece.parent_key], chosen_piece.child_key, new_node_name)
            tree=update_parent_and_branch_length(tree, chosen_piece.child_key, chosen_piece.child_branch, new_node_name, u1)
            tree[regraft_key][regraft_branch]=new_node_name
        sibling_key=get_other_children(tree[parent_key], regraft_key)[0]
        grandfather_key=tree[parent_key][0]
        _=get_branch_length_and_reset(tree[sibling_key], parent_key, tree[parent_key][3], add=True)
        tree[sibling_key]=rename_parent(tree[sibling_key], parent_key, grandfather_key)
        if grandfather_key!='r':
            tree[grandfather_key]=_rename_child(tree[grandfather_key], parent_key, sibling_key)
        del tree[parent_key]

    return tree

def graft_onto_rooted_admixture(tree, insertion_spot, remove_key, root_key, remove_branch=0):
    tree[remove_key][remove_branch]='r'
    tree[root_key[0]],_=get_branch_length_and_reset(tree[root_key[0]], 'r', insertion_spot)
    return tree

def change_admixture(node):
    new_node=[node[1],node[0],1.0-node[2],node[4],node[3]]+node[5:]
    return new_node

def readjust_length(node):
    multip= 1.0/((1.0-node[2])**2+node[2]**2)
    node[3]=node[3]*multip
    return node, multip

def get_number_of_admixes(tree):
    return sum((1 for node in list(tree.values()) if node_is_admixture(node)))

def get_number_of_leaves(tree):
    return sum((1 for node in list(tree.values()) if node_is_leaf_node(node)))

def remove_admix2(tree, rkey, rbranch, pks={}):
    '''
    removes an admixture. besides the smaller tree, also the disappeared branch lengths are returned.
    parent_key          sparent_key
        |                |
        |t_1             | t_4
        |   __---- source_key
      rkey/   t_5       \
        |                \t_3
        |t_2          sorphanota_key
    orphanota_key
    and alpha=admixture proportion. The function returns new_tree, (t1,t2,t3,t4,t5), alpha

    Note that t_4 could be None if source key is root. The source_key node is not an admixture node by assumption.
    '''
    rnode= tree[rkey]
    orphanota_key= get_children(rnode)[0]
    parent_key= rnode[other_branch(rbranch)]
    source_key= rnode[rbranch]
    t1= rnode[3+other_branch(rbranch)]
    t5= rnode[3+rbranch]
    if rbranch == 0:
        alpha=1-rnode[2]
    else:
        alpha=rnode[2]

    tree[orphanota_key],t2=get_branch_length_and_reset(tree[orphanota_key], rkey, t1, add=True)
    tree[orphanota_key]=_update_parent(tree[orphanota_key], rkey, parent_key)
    orphanota_branch=_get_index_of_parent(tree[orphanota_key], parent_key)
    pks['orphanota_key']=orphanota_key
    pks['orphanota_branch']=orphanota_branch

    if parent_key!='r':
        tree[parent_key]=_update_child(tree[parent_key], old_child=rkey, new_child=orphanota_key)
    if source_key=='r':
        tree,_,t3,_=remove_root_attachment(tree, rkey, 0) #now sorphanota_key is new root
        del tree[rkey]
        return tree, (t1,t2,t3,None,t5),alpha
    del tree[rkey]
    source_node=tree[source_key]
    sorphanota_key=get_other_children(source_node, child_key=rkey)[0]
    sparent_key=source_node[0]
    t4=source_node[3]
    if sparent_key!='r':
        tree[sparent_key]=_update_child(tree[sparent_key], source_key, sorphanota_key)
    tree[sorphanota_key],t3=get_branch_length_and_reset(tree[sorphanota_key], source_key, t4, add=True)
    tree[sorphanota_key]=_update_parent(tree[sorphanota_key], source_key, sparent_key)
    del tree[source_key]
    return tree, (t1,t2,t3,t4,t5), alpha

def direct_all_admixtures(tree, smaller_than_half=True):
    for key, node in list(tree.items()):
        if node_is_admixture(node):
            alpha=get_admixture_proportion_from_key(tree, key)
            if (smaller_than_half and alpha<0.5) or (not smaller_than_half and alpha>0.5):
                tree[key]=change_admixture(node)
    return tree

def other_branch(branch):
    if branch==0:
        return 1
    elif branch==1:
        return 0
    else:
        assert False, 'illegal branch'

def _update_parents(node, new_parents):
    if len(new_parents)==1:
        res=node[:5]+[new_parents[0],None]
        return res
    if len(new_parents)==2:
        res=node[:5]+new_parents
        return res
    if len(new_parents)==0:
        res=node[:5]+[None]*2
        return res

def _update_parent(node, old_parent, new_parent):
    if node[0]==old_parent:
        node[0]=new_parent
    elif node[1]==old_parent:
        node[1]=new_parent
    return node

def _update_child(node, old_child, new_child):
    if node[5]==old_child:
        node[5]=new_child
    elif node[6]==old_child:
        node[6]=new_child
    return node

def remove_non_mixing_admixtures(tree, limit=1e-7):
    '''
    Trees can have admixture events with admixture proportion very close to 0 or 1. This will remove those admixture proportions.
    '''
    admixtures_to_remove=[]
    for key, node in list(tree.items()):
        if node_is_admixture(node):
            if node[2]<limit:
                admixtures_to_remove.append((key,0))
            elif node[2]>1.0-limit:
                admixtures_to_remove.append((key,1))
    for adm_key, adm_branch in admixtures_to_remove:
        if adm_key in tree:#it could have been removed by others
            tree=remove_admixture(tree, adm_key, adm_branch)
    return tree

def get_parents(node):
    return node[:2]

def insert_admixture_node_halfly(tree, source_key, source_branch, insertion_spot, admix_b_length, new_node_name, admixture_proportion=0.51):
    '''
    Source should be sink, I guess in hindsight.
    '''
    node=tree[source_key]
    old_parent=node[source_branch]
    old_branch_length=node[source_branch+3]
    node[source_branch]=new_node_name
    node[source_branch+3]=old_branch_length*insertion_spot
    tree[new_node_name]=[old_parent, None, admixture_proportion, old_branch_length*(1-insertion_spot), admix_b_length, source_key, None]
    if old_parent != 'r':
        tree[old_parent]=_update_child(tree[old_parent], old_child=source_key, new_child=new_node_name)
    return tree

def get_real_parents(node):
    ps=node[:2]
    return [p for p in ps if p is not None]

def get_real_children(node):
    cs=node[5:7]
    return [c for c in cs if c is not None]

def get_keys_and_branches_from_children(tree, key):
    '''
    the key has to be an admixture
    '''

    child_keys=get_real_children(tree[key])
    branches=[]
    for child_key in child_keys:
        branches.append(mother_or_father(tree, child_key, key))
    return list(zip(child_keys, branches))

def get_other_parent(node, parent_key):
    if parent_key==node[0]:
        return node[1]
    elif parent_key==node[1]:
        return node[0]
    else:
        assert False, 'the shared parent was not a parent of the sibling.'

def halfbrother_is_uncle(tree, key, parent_key):
    '''
    parent_key is a non admixture node. This function checks if a sibling is an admixture that goes to the parent and the grandparent at the same time.
    If removed, there would be a loop where one person has two of the same parent.
    '''
    sibling_key=get_other_children(tree[parent_key], key)[0]
    if node_is_non_admixture(tree[sibling_key]):
        return False
    bonus_parent=get_other_parent(tree[sibling_key], parent_key)
    grand_parent_key=tree[parent_key][0]
    return bonus_parent==grand_parent_key

def remove_admixture(tree, key, branch):
    parent_key=tree[key][branch]

    while parent_key!='r' and node_is_admixture(tree[parent_key]):
        tree=remove_admixture(tree, parent_key, 1)
        parent_key=tree[key][branch]

    return remove_admix2(tree, key,branch)[0]

def scale_tree_copy(tree, factor):
    cop=deepcopy(tree)
    for key,node in list(cop.items()):
        node[3]*=factor
        if node_is_admixture(node):
            node[4]*=factor
        cop[key]=node
    return cop

def parent_is_spouse(tree, key, direction):
    '''
    key is an admixture node. This function checks if the parent in the direction of 'direction' also has a child with the admixture node.
    '''
    parent_key=tree[key][direction]
    offspring_key=get_real_children(tree[key])[0] #there is only one because key is an admixture node
    if node_is_non_admixture(tree[offspring_key]):
        return False
    spouse_key=get_other_parent(tree[offspring_key], key)
    return spouse_key == parent_key

def parent_is_sibling(tree, key, direction):
    '''
    key is an admixture node. This function checks if the parent in the direction of 'direction' is also the child of the parent in the direction of
    'other_branch(direction)'.
    '''
    parent_key=tree[key][direction]
    other_parent_key=tree[key][other_branch(direction)] #there is only one because key is an admixture node
    return (parent_key=='r' or other_parent_key in get_real_parents(tree[parent_key]))

def get_all_admixture_origins(tree):
    res={}
    for key,node in list(tree.items()):
        if node_is_admixture(node):
            res[key]=(node[3], get_parents(node)[0])
    return res

def is_root(*keys):
    ad=[key=='r' for key in list(keys)]
    return any(ad)

def to_networkx_format(tree):
    edges=[]
    admixture_nodes=[]
    leaves=[]
    root=['r']
    coalescence_nodes=[]
    edge_lengths=[]
    for key,node in list(tree.items()):
        if node_is_leaf_node(node):
            leaves.append(key)
        else:
            if node_is_coalescence(node):
                coalescence_nodes.append(key)
            if node_is_admixture(node):
                admixture_nodes.append(key)
        ps=get_real_parents(node)
        for p in ps:
            edges.append((p,key))
            branch=mother_or_father(tree, key, p)
            edge_lengths.append(get_branch_length(tree, key, branch))
    return leaves, admixture_nodes, coalescence_nodes, root, edges, edge_lengths


class Covariance_Matrix():

    def __init__(self, nodes_to_index):
        self.ni=nodes_to_index
        self.covmatr=zeros((len(nodes_to_index), len(nodes_to_index)))

    def get_indices(self, nodes):
        return [self.ni[n] for n in nodes]

    def get_addon(self, branch_length, weights):
        return branch_length*outer(weights, weights)

    def update(self, branch_length, population):
        indices=self.get_indices(population.members)
        self.covmatr[ix_(indices,indices)]+=self.get_addon(branch_length, population.weights)

class Population:

    def __init__(self, weights,members):
        self.weights=weights
        self.members=members

    def get_weight(self, member):
        for m,w in zip(self.members, self.weights):
            if m==member:
                return w

    def subset_of_the_candidates(self,candidates):
        if any(( (cand in self.members) and (self.get_weight(cand)>1e-7) for cand in candidates)):
            if any((   (cand not in self.members) or (self.get_weight(cand)<1.0-1e-7) for cand in candidates)):
                return 'partly'
            else:
                return 'all'
        return 'none'

    def get_population_string(self, keys_to_remove=[]):
        return '.'.join(sorted([m for m,w in zip(self.members,self.weights) if w>0.0 and m not in keys_to_remove]))

    def remove_partition(self, weight):
        n_w=[w*weight for w in self.weights]
        self.weights=[w*(1-weight) for w in self.weights]
        return Population(n_w, deepcopy(self.members))

    def merge_with_other(self, other):

        self.weights=[w+other.weights[other.members.index(m)] if m in other.members else w for m,w in zip(self.members,self.weights) ]
        tmpl=[(w,m) for w,m in zip(other.weights, other.members) if m not in self.members]
        if tmpl:
            x_w,x_m=list(zip(*tmpl))
            self.weights.extend(x_w)
            self.members.extend(x_m)
        return self

def leave_node(key, node, population, covmat):
    if node_is_non_admixture(node):
        return [follow_branch(parent_key=node[0],branch_length=node[3], population=population, covmat=covmat)]
    else:
        new_pop=population.remove_partition(1.0-node[2])
        return [follow_branch(parent_key=node[0],branch_length=node[3], population=population, covmat=covmat, dependent='none'), #changed dependent='none' to go to most loose restriction that still makes sense. To go back,put dependent=node[1
                follow_branch(parent_key=node[1],branch_length=node[4], population=new_pop, covmat=covmat, dependent='none')]

def follow_branch(parent_key, branch_length, population, covmat, dependent="none"):
    covmat.update(branch_length, population)
    return parent_key, population, dependent

def _add_to_waiting(dic,add,tree):
    key,pop,dep=add
    if key in dic:#this means that the node is a coalescence node
        dic[key][0][1]=pop
        dic[key][1][1]=dep
    else:
        if key=='r' or node_is_non_admixture(tree[key]):
            dic[key]=[[pop,None],[dep,"empty"]]
        else:
            dic[key]=[[pop],[dep]]
    return dic

def _full_node(key,dic):
    if key in dic:
        for dep in dic[key][1]:
            if dep=="empty":
                return False
        return True
    return False

def _merge_pops(pops):
    if len(pops)==1:
        return pops[0]
    else:
        return pops[0].merge_with_other(pops[1])

def _thin_out_dic(dic, taken):
    ready_nodes=[]
    for key,[pops, deps] in list(dic.items()):
        full_node=True
        for dep in deps:
            if dep is None or not (dep=="none" or _full_node(dep,dic) or dep in taken):
                full_node=False
            else:
                pass
        if full_node:
            taken.append(key)
            ready_nodes.append((key,_merge_pops(pops)))
            del dic[key]
    return dic,ready_nodes

def make_covariance(tree, node_keys=None, old_cov=False):
    if node_keys is None:
        node_keys=sorted(get_leaf_keys(tree))
    pops=[Population([1.0],[node]) for node in node_keys]
    ready_nodes=list(zip(node_keys,pops))
    covmat=Covariance_Matrix({node_key:n for n,node_key in enumerate(node_keys)})
    if old_cov:
        covmat=Covariance_Matrix({node_key:n for n,node_key in enumerate(node_keys)})
    waiting_nodes={}
    taken_nodes=[]
    while True:
        for key,pop in ready_nodes:
            upds=leave_node(key, tree[key], pop, covmat)
            for upd in upds:
                waiting_nodes=_add_to_waiting(waiting_nodes, upd,tree)
            taken_nodes.append(key)
        waiting_nodes,ready_nodes=_thin_out_dic(waiting_nodes, taken_nodes[:])
        if len(ready_nodes)==0:
            return None
        if len(ready_nodes)==1 and ready_nodes[0][0]=="r":
            break

    return covmat.covmatr

class dummy_covmat(object):

    def update(self, *args, **kwargs):
        pass

def get_populations(tree, keys_to_include=None):

    node_keys=sorted(get_leaf_keys(tree))
    if keys_to_include is None:
        keys_to_remove=[]
    else:
        keys_to_remove=list(set(node_keys)-set(keys_to_include))
    pops=[Population([1.0],[node]) for node in node_keys]
    ready_nodes=list(zip(node_keys,pops))
    waiting_nodes={}
    taken_nodes=[]
    covmat=dummy_covmat()
    pop_strings=[]
    while True:
        for key,pop in ready_nodes:
            pop_strings.append(pop.get_population_string( keys_to_remove))
            upds=leave_node(key, tree[key], pop, covmat)
            for upd in upds:
                waiting_nodes=_add_to_waiting(waiting_nodes, upd,tree)
            taken_nodes.append(key)
        waiting_nodes,ready_nodes=_thin_out_dic(waiting_nodes, taken_nodes[:])
        if len(ready_nodes)==0:
            return None
        if len(ready_nodes)==1 and ready_nodes[0][0]=="r":
            big_pop=ready_nodes[0][1]
            pop_strings.append(big_pop.get_population_string( keys_to_remove))
            break
    if '' in pop_strings:
        pop_strings.remove('')
    return sorted(list(set(pop_strings)))

def get_populations_string(tree, min_w=0.0, keys_to_include=None):
    return '-'.join(get_populations(tree, keys_to_include))

class uniform_prior(object):

    def __init__(self, leaves):
        self.leaves=leaves  # we simply precompute the number of graphs with the given topology
        self.dic_of_counts=[ [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ] ,  [ 1 , 0.0 , 4.0 , 66.0 , 1692.0 , 59396.0 , 2674940.0 , 147852644.0 , 9718273512.0 , 741990839512.0 , 64626724327952.0 , 6330275159852480.0 , 6.892947420189025e+17 , 8.264443160965759e+19 , 1.0823634468700586e+22 , 1.537907404982144e+24 , 2.3569237209445557e+26 , 3.8762655727514517e+28 , 6.810848608636099e+30 , 1.2735067083415438e+33 , 2.5251941914056145e+35 , 5.293249636742438e+37 , 1.1696615965656003e+40 , 2.7176972014602205e+42 , 6.6242752148630366e+44 ] ,  [ 1.0 , 4.0 , 70.0 , 1758.0 , 61088.0 , 2734336.0 , 150527584.0 , 9866126156.0 , 751709113024.0 , 65368715167464.0 , 6394901884180432.0 , 6.956250171787551e+17 , 8.333372635167649e+19 , 1.0906278900310244e+22 , 1.5487310394508445e+24 , 2.3723027949943773e+26 , 3.899834809960897e+28 , 6.849611264363614e+30 , 1.2803175569501799e+33 , 2.53792925848903e+35 , 5.318501578656494e+37 , 1.1749548462023429e+40 , 2.7293938174258765e+42 , 6.6514521868776385e+44 , 1.696883539592601e+47 ] ,  [ 3.0 , 42.0 , 1104.0 , 39402.0 , 1798200.0 , 100443612.0 , 6658258956.0 , 511901611812.0 , 44845333519416.0 , 4414291963659240.0 , 4.826986063565739e+17 , 5.808680328874313e+19 , 7.631950266115392e+21 , 1.0875057114759366e+24 , 1.670910066959595e+26 , 2.754327613879423e+28 , 4.849566910151839e+30 , 9.084924093349353e+32 , 1.8045250480428444e+35 , 3.7885882667636454e+37 , 8.383947858205278e+39 , 1.9506325249497383e+42 , 4.76054786721098e+44 , 1.2161265045787953e+47 , 3.2456585385859906e+49 ] ,  [ 15.0 , 456.0 , 18030.0 , 876330.0 , 51101220.0 , 3495342120.0 , 275246122128.0 , 24574461582300.0 , 2456510295056880.0 , 2.7207588213803533e+17 , 3.3096927244300124e+19 , 4.3890765923871527e+21 , 6.304720269749876e+23 , 9.75562843626133e+25 , 1.6182069273427536e+28 , 2.86513783746645e+30 , 5.3943902327180374e+32 , 1.0763509056151257e+35 , 2.269130780820234e+37 , 5.040427691590846e+39 , 1.1767834622158333e+42 , 2.88112779822478e+44 , 7.381846422023449e+46 , 1.9755031007324542e+49 , 5.5123870066829325e+51 ] ,  [ 105.0 , 5610.0 , 316980.0 , 20253390.0 , 1474122780.0 , 121456867140.0 , 11225696603100.0 , 1153322062373460.0 , 1.3061952420746509e+17 , 1.618672616750682e+19 , 2.180510440741351e+21 , 3.174678164988636e+23 , 4.970171851051007e+25 , 8.329378167321895e+27 , 1.4882556050192292e+30 , 2.8249098626778145e+32 , 5.677932831234502e+34 , 1.2049454712766194e+37 , 2.6926952976335928e+39 , 6.3212643172055014e+41 , 1.5554715667173788e+44 , 4.0039202329633095e+46 , 1.0761378588320843e+49 , 3.0148500212125966e+51 , 8.79003102775789e+53 ] ,  [ 945.0 , 78660.0 , 6061230.0 , 494434710.0 , 44033580360.0 , 4308205597920.0 , 462384938927520.0 , 5.421670148297718e+16 , 6.911653409447362e+18 , 9.532994743052194e+20 , 1.415984678589116e+23 , 2.255278944892307e+25 , 3.8365580113960213e+27 , 6.945785450397294e+29 , 1.333880614482516e+32 , 2.7091490657814488e+34 , 5.803458131229215e+36 , 1.3079694242263177e+39 , 3.094382041001295e+41 , 7.668402928916149e+43 , 1.986786439110278e+46 , 5.372016132382956e+48 , 1.5133702399277173e+51 , 4.435148969493036e+53 , 1.3502249164138617e+56 ] ,  [ 10395.0 , 1246770.0 , 126148680.0 , 12822742170.0 , 1373863821120.0 , 157642585527420.0 , 1.946132343270486e+16 , 2.584990286105592e+18 , 3.687272260347718e+20 , 5.632889234094434e+22 , 9.188293873845162e+24 , 1.5955212797861172e+27 , 2.9407777197755668e+29 , 5.737292947516349e+31 , 1.1816946039766288e+34 , 2.56331204272437e+36 , 5.842711244332498e+38 , 1.3964736625759083e+41 , 3.493092538517179e+43 , 9.127669525825847e+45 , 2.487427671251343e+48 , 7.058299492849912e+50 , 2.0824414092109864e+53 , 6.379318696096029e+55 , 2.0265082476780733e+58 ] ,  [ 135135.0 , 22110480.0 , 2849626710.0 , 353715686730.0 , 44937162049500.0 , 5981644214736120.0 , 8.422425184760344e+17 , 1.2587992880809514e+20 , 1.99813749683552e+22 , 3.3656530947734854e+24 , 6.0062889856452236e+26 , 1.1334690537153896e+29 , 2.257342721938297e+31 , 4.734564552428599e+33 , 1.0437260030928995e+36 , 2.413676007621082e+38 , 5.844659940856465e+40 , 1.4793519408974045e+43 , 3.907532352863126e+45 , 1.0754268328121754e+48 , 3.07945926143974e+50 , 9.16197194943249e+52 , 2.8285578843184983e+55 , 9.050577152989247e+57 , 2.99796456259025e+60 ] ,  [ 2027025.0 , 434475090.0 , 69590461500.0 , 10370332504350.0 , 1542747891554100.0 , 2.359554693064569e+17 , 3.760796421221244e+19 , 6.286119119648026e+21 , 1.1048365045490366e+24 , 2.0433631915356808e+26 , 3.975423893493063e+28 , 8.128576707183892e+30 , 1.7446292521043394e+33 , 3.9250351800309506e+35 , 9.242731418790313e+37 , 2.274746039374392e+40 , 5.842653554665628e+42 , 1.5639352465581557e+45 , 4.3568071710580135e+47 , 1.2615182567931965e+50 , 3.79188502202146e+52 , 1.1817893799512006e+55 , 3.814702791507353e+57 , 1.273954809336227e+60 , 4.397243275160826e+62 ] ,  [ 34459425.0 , 9378690300.0 , 1829453991750.0 , 322593910773750.0 , 5.5587842129358e+16 , 9.686095821672454e+18 , 1.73573399904912e+21 , 3.226730137332095e+23 , 6.250682645775979e+25 , 1.2644406160279983e+28 , 2.673051150097343e+30 , 5.905224810309482e+32 , 1.3625719072849394e+35 , 3.2811453616557544e+37 , 8.237814934026635e+39 , 2.15406110677199e+42 , 5.859829393039597e+44 , 1.6565733008318603e+47 , 4.861359978941048e+49 , 1.4793109771881837e+52 , 4.662970744626854e+54 , 1.5210031151986702e+57 , 5.129099961364211e+59 , 1.7864564634808462e+62 , 6.420924455081439e+64 ] ,  [ 654729075.0 , 220735226250.0 , 51558524740800.0 , 1.062501750501345e+16 , 2.1005378089249551e+18 , 4.138590557336033e+20 , 8.287309910735932e+22 , 1.70481314877327e+25 , 3.624552208013228e+27 , 7.990816047584342e+29 , 1.8299346891885265e+32 , 4.356195872147052e+34 , 1.07809422901366e+37 , 2.773081313522608e+39 , 7.409612871456526e+41 , 2.0552184408834505e+44 , 5.913048594389061e+46 , 1.7631700550824565e+49 , 5.444193158264257e+51 , 1.7392173270153975e+54 , 5.743574889275701e+56 , 1.9590805036247796e+59 , 6.896139684352723e+61 , 2.5032030729229716e+64 , 9.36237554912532e+66 ] ,  [ 13749310575.0 , 5627628883800.0 , 1551643265922750.0 , 3.696898548922078e+17 , 8.314938966258935e+19 , 1.8399179790515601e+22 , 4.094308480766523e+24 , 9.27733409605142e+26 , 2.1563573973377658e+29 , 5.16389033343759e+31 , 1.27734758824016e+34 , 3.2684158061749276e+36 , 8.656857691670416e+38 , 2.373913045102592e+41 , 6.739020841418545e+43 , 1.9797405660691056e+46 , 6.015740726690857e+48 , 1.8896818935679622e+51 , 6.132408915801482e+53 , 2.0545917617187702e+56 , 7.101887734494919e+58 , 2.5309028161280987e+61 , 9.29247869397658e+63 , 3.512749298866248e+66 , 1.3662589326596516e+69 ] ,  [ 316234143225.0 , 154540438702650.0 , 4.9685512095519304e+16 , 1.3558219850832808e+19 , 3.443481454435169e+21 , 8.505790999185392e+23 , 2.092819384798582e+26 , 5.201353222357435e+28 , 1.316952216386522e+31 , 3.4150778370540035e+33 , 9.100234372005574e+35 , 2.4969814274571685e+38 , 7.063359938361477e+40 , 2.0611791846727972e+43 , 6.206322543530538e+45 , 1.9282176555719457e+48 , 6.1800055368172525e+50 , 2.0426069244267094e+53 , 6.959172579419408e+55 , 2.4428418535449347e+58 , 8.830178972206446e+60 , 3.2850464606181532e+63 , 1.25709352940651e+66 , 4.945408593935559e+68 , 1.9989434049574286e+71 ] ,  [ 7905853580625.0 , 4548496183931700.0 , 1.6872468936770947e+18 , 5.2296971915687284e+20 , 1.4898560942312324e+23 , 4.085579592444959e+25 , 1.10639906961173e+28 , 3.004338281669421e+30 , 8.258677976702356e+32 , 2.3123953414723115e+35 , 6.62124830753406e+37 , 1.9439045779289892e+40 , 5.861276062981373e+42 , 1.8169284896889523e+45 , 5.7938038149710755e+47 , 1.901010925490097e+50 , 6.418257751474692e+52 , 2.2295045604002117e+55 , 7.96627428572811e+57 , 2.9269770991600815e+60 , 1.1054449571370305e+63 , 4.289711087383059e+65 , 1.7096220249301006e+68 , 6.994461125160964e+70 , 2.936213492187441e+73 ] ,  [ 213458046676875.0 , 1.4285862691825726e+17 , 6.057961759634886e+19 , 2.117161063127044e+22 , 6.724943600109133e+24 , 2.037163461540319e+27 , 6.046444240201144e+29 , 1.787440342258185e+32 , 5.318109821464104e+34 , 1.6034664955996656e+37 , 4.921947867806675e+39 , 1.5428989255710436e+42 , 4.949594551949644e+44 , 1.6271803894913636e+47 , 5.486827849839531e+49 , 1.898701379966575e+52 , 6.744644226258608e+54 , 2.459593825680263e+57 , 9.207551025276588e+59 , 3.537783450568064e+62 , 1.3948327798268677e+65 , 5.641447587433045e+67 , 2.3398779275653558e+70 , 9.948894916682057e+72 , 4.334830908415566e+75 ] ,  [ 6190283353629375.0 , 4.769565128700372e+18 , 2.2933745370302306e+21 , 8.977988118149253e+23 , 3.16248343515694e+26 , 1.0534672603312672e+29 , 3.413777140680935e+31 , 1.0950162753198785e+34 , 3.516091851269476e+36 , 1.1387274890623734e+39 , 3.738796165940563e+41 , 1.2489407963901956e+44 , 4.255250153589741e+46 , 1.4812429220307178e+49 , 5.274166280001371e+51 , 1.9224024948559302e+54 , 7.176424553039516e+56 , 2.744490871801571e+59 , 1.0753549261154093e+62 , 4.316901784172634e+64 , 1.7753223288893567e+67 , 7.478107447951108e+69 , 3.2256568431960323e+72 , 1.4244351723611598e+75 , 6.437827400773274e+77 ] ,  [ 1.9189878396251066e+17 , 1.6868977303053805e+20 , 9.13120737054223e+22 , 3.980589449774141e+25 , 1.5473101573208855e+28 , 5.644450458290238e+30 , 1.9899014779655654e+33 , 6.904550105534191e+35 , 2.386291693708917e+38 , 8.281619105943941e+40 , 2.902382156416155e+43 , 1.0312542752480833e+46 , 3.7254253675756906e+48 , 1.3710718231270755e+51 , 5.148018261884774e+53 , 1.974015027069998e+56 , 7.735489687085645e+58 , 3.0991644523396176e+61 , 1.2697979065101877e+64 , 5.321223417737698e+66 , 2.280778553023606e+69 , 9.998207552204631e+71 , 4.4820249897866705e+74 , 2.0543033683926486e+77 , 9.625038242433284e+79 ] ,  [ 6.332659870762852e+18 , 6.300807584446831e+21 , 3.814894187061798e+24 , 1.8420640717629812e+27 , 7.866324755477309e+29 , 3.1304854732156613e+32 , 1.1967145410284336e+35 , 4.478902564576687e+37 , 1.6619564665721288e+40 , 6.167110631857386e+42 , 2.302466422901017e+45 , 8.686421071207564e+47 , 3.3219301146673405e+50 , 1.290735510974484e+53 , 5.103951623030923e+55 , 2.0564790827945498e+58 , 8.45019925640291e+60 , 3.5432162373165665e+63 , 1.5166763177862636e+66 , 6.629219669939956e+68 , 2.9591307804718796e+71 , 1.3490122903241978e+74 , 6.280641992072466e+76 , 2.9860046294372345e+79 , 1.4494995368227643e+82 ] ,  [ 2.216430954766998e+20 , 2.4785312583709413e+23 , 1.6688471330728577e+26 , 8.882698031980977e+28 , 4.150226268058308e+31 , 1.7954861439688e+34 , 7.42006724756413e+36 , 2.9875137775216166e+39 , 1.1874050531290512e+42 , 4.7014012470390825e+44 , 1.8663976506226433e+47 , 7.463869359416944e+49 , 3.0171726923806325e+52 , 1.235998478290066e+55 , 5.140895923032812e+57 , 2.1740675318492005e+60 , 9.357749035769276e+62 , 4.102635722937248e+65 , 1.8330862379192958e+68 , 8.350133454394689e+70 , 3.8788200278362065e+73 , 1.8376455867204847e+76 , 8.879828626122951e+78 , 4.376479211790723e+81 , 2.1998707615298404e+84 ] ,  [ 8.200794532637891e+21 , 1.0242156814789283e+25 , 7.629265056309208e+27 , 4.456634110205921e+30 , 2.2696674251888822e+33 , 1.0639816489987605e+36 , 4.7400032768622575e+38 , 2.0479638459485098e+41 , 8.699547821725604e+43 , 3.6680923595322604e+46 , 1.5456812111836973e+49 , 6.541924760767583e+51 , 2.791304846900022e+54 , 1.2040154547560529e+57 , 5.26126680630474e+59 , 2.3327626933305157e+62 , 1.0507340538452558e+65 , 4.812244319762026e+67 , 2.2424629315958e+70 , 1.0637439170719607e+73 , 5.13845509742169e+75 , 2.5282030920568416e+78 , 1.2671742218222549e+81 , 6.470463667040849e+83 , 3.365997447306438e+86 ] ,  [ 3.1983098677287775e+23 , 4.4360348766561075e+26 , 3.6383047310006335e+29 , 2.3231232214619353e+32 , 1.2851489808586776e+35 , 6.508513788252129e+37 , 3.117462748337807e+40 , 1.4420374846100435e+43 , 6.533391292978231e+45 , 2.9281597206994606e+48 , 1.3075423546580956e+51 , 5.848110277864854e+53 , 2.630218446147541e+56 , 1.1931234833566147e+59 , 5.471282170654846e+61 , 2.54076357012055e+64 , 1.19645241492517e+67 , 5.719141009182526e+69 , 2.777231131404615e+72 , 1.3708718442540269e+75 , 6.881395190491022e+77 , 3.513896723674372e+80 , 1.825701189709968e+83 , 9.652939124668551e+85 , 5.194116844263601e+88 ] ,  [ 1.3113070457687988e+25 , 2.009552855742514e+28 , 1.8069358233348818e+31 , 1.2564958134609054e+34 , 7.526279230127273e+36 , 4.106319196044759e+39 , 2.1094996110423233e+42 , 1.0424124404121198e+45 , 5.027413562463226e+47 , 2.390865751505517e+50 , 1.1295721948742096e+53 , 5.33124694237315e+55 , 2.524157633686647e+58 , 1.2027205336128905e+61 , 5.781532041498206e+63 , 2.809183883536353e+66 , 1.3817241497431199e+69 , 6.887609832625254e+71 , 3.4826653641804657e+74 , 1.7875257117067417e+77 , 9.317958085724306e+79 , 4.935041888996905e+82 , 2.6563811582108555e+85 , 1.4534829869214144e+88 , 8.085552816351555e+90 ] ,  [ 5.638620296805836e+26 , 9.503319670068634e+29 , 9.331331419429265e+32 , 7.042548010090945e+35 , 4.554085750791883e+38 , 2.6698603491867056e+41 , 1.4676545977429565e+44 , 7.73178216826518e+46 , 3.9621523605226544e+49 , 1.9960863296043551e+52 , 9.962939701485944e+54 , 4.955281797070667e+57 , 2.4667798315944206e+60 , 1.2332237110290713e+63 , 6.207880742025678e+65 , 3.153023296605044e+68 , 1.6184327255954398e+71 , 8.40617410481652e+73 , 4.422560985749132e+76 , 2.3586491548755204e+79 , 1.2759528860853505e+82 , 7.004791131881917e+84 , 3.903943901922612e+87 , 2.209420712645152e+90 , 1.2700056944424907e+93 ] ,  [ 2.5373791335626256e+28 , 4.683387640043951e+31 , 5.003561195634751e+34 , 4.0857167578808044e+37 , 2.844441879855096e+40 , 1.787497548624223e+43 , 1.04918760154045e+46 , 5.881225136151547e+48 , 3.1967893265286455e+51 , 1.703416151214019e+54 , 8.96938091741423e+56 , 4.695139694889786e+59 , 2.4545521101028983e+62 , 1.2861060382654852e+65 , 6.772811459533379e+67 , 3.5925287256301844e+70 , 1.922758531878153e+73 , 1.0397825567985085e+76 , 5.687636349484167e+78 , 3.1497323624879685e+81 , 1.7671394924262226e+84 , 1.0049924082680283e+87 , 5.79611798625518e+89 , 3.3910911555886446e+92 , 2.0131675069476688e+95 ] ,  [ 1.192568192774434e+30 , 2.4013222296606967e+33 , 2.7820816890671186e+36 , 2.4507369436126353e+39 , 1.8321887318755146e+42 , 1.231384562763782e+45 , 7.701840482556801e+47 , 4.585423569763425e+50 , 2.639425268360347e+53 , 1.485356894537822e+56 , 8.239871577233039e+58 , 4.533953423701398e+61 , 2.4864096929631322e+64 , 1.3640178819321102e+67 , 7.5073726086921e+69 , 4.155114972475837e+72 , 2.316909221412885e+75 , 1.3034985112967272e+78 , 7.408087895369114e+80 , 4.2570947198049146e+83 , 2.4755294601159125e+86 , 1.4576060471687818e+89 , 8.694505374330877e+91 , 5.255970621332592e+94 , 3.221053875241414e+97 ] ,  [ 5.843584144594727e+31 , 1.2790741969754552e+35 , 1.6020448970525406e+38 , 1.518321306334275e+41 , 1.2160265667226753e+44 , 8.722026362750464e+46 , 5.802084142791104e+49 , 3.6626443610283815e+52 , 2.2291287539615414e+55 , 1.3230001708812596e+58 , 7.722180894162086e+60 , 4.4612544458970285e+63 , 2.5636346231316187e+66 , 1.4710077719404735e+69 , 8.453959074736178e+71 , 4.878092782258933e+74 , 2.831620047392828e+77 , 1.6561570991874985e+80 , 9.77251219150208e+82 , 5.823727947625584e+85 , 3.5079510001392025e+88 , 2.1372823808019152e+91 , 1.317855788116093e+94 , 8.227468939178388e+96 , 5.202509224517763e+99 ] ,  [ 2.980227913743311e+33 , 7.067932971009596e+36 , 9.543063465291988e+39 , 9.706077615178908e+42 , 8.309157444919429e+45 , 6.347665040729473e+48 , 4.48297867006613e+51 , 2.9957065526318805e+54 , 1.9249161830035766e+57 , 1.2032575962300993e+60 , 7.380705127567628e+62 , 4.471858263307846e+65 , 2.689938503608543e+68 , 1.6128685274217198e+71 , 9.670270659563147e+73 , 5.81257156053834e+76 , 3.509797259699217e+79 , 2.1325830503984668e+82 , 1.3056670753996852e+85 , 8.063919497916501e+87 , 5.028567934164281e+90 , 3.168490853720728e+93 , 2.0185392841845948e+96 , 1.3008172879890812e+99 , 8.483337556379322e+101 ] ,  [ 1.5795207942839545e+35 , 4.046495628245275e+38 , 5.874018769676728e+41 , 6.396397145783996e+44 , 5.840783511431729e+47 , 4.743444848869359e+50 , 3.5505562766107182e+53 , 2.507750562777709e+56 , 1.6988810545916334e+59 , 1.1170652911195612e+62 , 7.192338890570887e+64 , 4.565282295885866e+67 , 2.8717570691433584e+70 , 1.797649750152498e+73 , 1.1234952610399573e+76 , 7.029084912005404e+78 , 4.411872038067991e+81 , 2.782960670546624e+84 , 1.7667599134069565e+87 , 1.1301858109971547e+90 , 7.292046154553795e+92 , 4.7492787333008745e+95 , 3.124459947684429e+98 , 2.0774515864522693e+101 , 1.3966663920267587e+104 ] ,  [ 8.68736436856175e+36 , 2.3973541777199173e+40 , 3.7322638344768385e+43 , 4.3416947981767247e+46 , 4.2204738382722867e+49 , 3.6373038921552806e+52 , 2.8809513144223984e+55 , 2.1475693226643262e+58 , 1.5318483144429747e+61 , 1.0582127007863553e+64 , 7.143859346462981e+66 , 4.745609058273807e+69 , 3.118789391024358e+72 , 2.0363991299537222e+75 , 1.3255666753323387e+78 , 8.625763843052199e+80 , 5.623735206929368e+83 , 3.680291053560562e+86 , 2.4211839860230363e+89 , 1.6032695636119627e+92 , 1.0697212322412808e+95 , 7.197737791436338e+97 , 4.887600841587259e+100 , 3.351430171413132e+103 , 2.3217517238094852e+106 ] ,  [ 4.951797690080198e+38 , 1.468123292772432e+42 , 2.445571909906898e+45 , 3.0328884066916445e+48 , 3.132699184002e+51 , 2.8602602749354238e+54 , 2.3936272108604776e+57 , 1.880576755324636e+60 , 1.4105913396629157e+63 , 1.0225797081829016e+66 , 7.230407737101315e+68 , 5.02176642874355e+71 , 3.444844374869823e+74 , 2.344226042898184e+77 , 1.5880709883966686e+80 , 1.0740308745195702e+83 , 7.26859817795564e+85 , 4.931778615210711e+88 , 3.360192664500619e+91 , 2.301998510689104e+94 , 1.5874636054489986e+97 , 1.102960099883918e+100 , 7.726941280949159e+102 , 5.461695888185952e+105 , 3.897214554720126e+108 ] ,  [ 2.9215606371473166e+40 , 9.283501174692208e+43 , 1.6510700176200797e+47 , 2.1786467004713415e+50 , 2.3869843696290123e+53 , 2.3052459624430183e+56 , 2.035343851075399e+59 , 1.6831590972457458e+62 , 1.3260331884863899e+65 , 1.0076481631468686e+68 , 7.454827132921542e+70 , 5.408249793995362e+73 , 3.8690964019368246e+76 , 2.7418268336639245e+79 , 1.9315856845449623e+82 , 1.3567673118839328e+85 , 9.524877331936248e+87 , 6.69637099669868e+90 , 4.722404261490744e+93 , 3.3452552568298044e+96 , 2.3830854836045297e+99 , 1.7088970664930896e+102 , 1.2345626658554456e+105 , 8.991468065668529e+107 , 6.605727883224886e+110 ] ,  [ 1.782151988659863e+42 , 6.055498479077247e+45 , 1.1475076717274963e+49 , 1.6081550168143113e+52 , 1.8658364833113776e+55 , 1.9031404607831076e+58 , 1.7703836673519328e+61 , 1.539092235146927e+64 , 1.272080330499216e+67 , 1.0122054052671334e+70 , 7.827741102656446e+72 , 5.926366345827635e+75 , 4.417906719913438e+78 , 3.2576800258184253e+81 , 2.384896690505311e+84 , 1.7386407700721883e+87 , 1.2653397888288687e+90 , 9.212018094313406e+92 , 6.72043591045823e+95 , 4.9199318023695994e+98 , 3.6187995500106557e+101 , 2.6770510240934277e+104 , 1.9934716262388127e+107 , 1.4953476004750136e+110 , 1.1306342143402172e+113 ] ,  [ 1.1227557528557138e+44 , 4.07073087587946e+47 , 8.203502233046663e+50 , 1.2189151349470184e+54 , 1.4952825956005743e+57 , 1.6085420938773798e+60 , 1.5744929387591796e+63 , 1.4372438587105465e+66 , 1.2448731195466153e+69 , 1.036190662277655e+72 , 8.368359355857871e+74 , 6.606148600778574e+77 , 5.127446624220517e+80 , 3.931224089095857e+83 , 2.988622522424817e+86 , 2.2598247358884998e+89 , 1.703914056778973e+92 , 1.2838415129673413e+95 , 9.683594223897186e+97 , 7.322694610197642e+100 , 5.5585456115042035e+103 , 4.24004461101504e+106 , 3.253059351320211e+109 , 2.5122368319784366e+112 , 1.9541660030273985e+115 ] ,  [ 7.29791239356214e+45 , 2.817722799874254e+49 , 6.02787129272319e+52 , 9.480523470750214e+55 , 1.2278442451010058e+59 , 1.3911620439406547e+62 , 1.431069035729691e+65 , 1.3700905297485487e+68 , 1.2423206803592049e+71 , 1.0806579676629281e+74 , 9.106091415502678e+76 , 7.489173044698224e+79 , 6.0474763844858166e+82 , 4.8174901368879314e+85 , 3.800597172673105e+88 , 2.9788120863252953e+91 , 2.3255877892662718e+94 , 1.8124628908287822e+97 , 1.412692102081623e+100 , 1.1029054873966146e+103 , 8.635929406313885e+105 , 6.789569776511665e+108 , 5.36474373866179e+111 , 4.263640194651569e+114 , 3.4106535014657295e+117 ] ,  [ 4.889601303686633e+47 , 2.006633169037784e+51 , 4.5491929203776276e+54 , 7.561837242563745e+57 , 1.0324986523380715e+61 , 1.2305277855760918e+64 , 1.3287306545477584e+67 , 1.3327639742580662e+70 , 1.263840503009186e+73 , 1.1478462823534807e+76 , 1.0083154039250716e+79 , 8.632645396034146e+81 , 7.2468115857687165e+84 , 5.99390850906047e+87 , 4.9039342639681125e+90 , 3.9815927867815614e+93 , 3.2167150893058357e+96 , 2.591706183893113e+99 , 2.0863876916141744e+102 , 1.680862858600852e+105 , 1.3570164126135554e+108 , 1.0991431796222182e+111 , 8.94062803281493e+113 , 7.309598332426275e+116 , 6.010991585171746e+119 ] ,  [ 3.373824899543777e+49 , 1.4690674209839198e+53 , 3.523800120910642e+56 , 6.1815081147854965e+59 , 8.886452868726398e+62 , 1.1126711045201484e+66 , 1.2597565059012843e+69 , 1.3224521374792304e+72 , 1.3102577714686655e+75 , 1.2413621421224616e+78 , 1.1358489646552713e+81 , 1.0115297959735267e+84 , 8.821279466746496e+86 , 7.570391995155453e+89 , 6.419214064152749e+92 , 5.39581573807142e+95 , 4.508544575878851e+98 , 3.753338173965157e+101 , 3.119191507598617e+104 , 2.5919218067922733e+107 , 2.1565711489872926e+110 , 1.798813430568247e+113 , 1.5056826836648719e+116 , 1.2658671039132191e+119 , 1.069743422054899e+122 ] ,  [ 2.3954156786760817e+51 , 1.1048359675503038e+55 , 2.7996929452353234e+58 , 5.175851676620348e+61 , 7.824140008126798e+64 , 1.0280287084980258e+68 , 1.219081845405603e+71 , 1.338049483887127e+74 , 1.3838441828415362e+77 , 1.3664950094835316e+80 , 1.3013493289100062e+83 , 1.2045919854081322e+86 , 1.0905384792782936e+89 , 9.704400735526126e+91 , 8.52306486439404e+94 , 7.412838801977041e+97 , 6.402539848464619e+100 , 5.504527996197576e+103 , 4.720088123704928e+106 , 4.043658341507398e+109 , 3.465914101426195e+112 , 2.975879554822754e+115 , 2.5622842963883136e+118 , 2.2143718693567994e+121 , 1.92233094985356e+124 ] ,  [ 1.7486534454335394e+53 , 8.529651015235261e+56 , 2.280138751950851e+60 , 4.436587037717678e+63 , 7.043684118030473e+66 , 9.70098352993213e+69 , 1.2036639917401079e+73 , 1.379995450970743e+76 , 1.4884958901190463e+79 , 1.530703110883567e+82 , 1.5160305589891252e+85 , 1.4575758776415043e+88 , 1.3689552159779188e+91 , 1.2623646708282772e+94 , 1.1476748014899845e+97 , 1.0322372678400603e+100 , 9.211025329696965e+102 , 8.174236111966996e+105 , 7.229021788559836e+108 , 6.381986020927992e+111 , 5.632719536473793e+114 , 4.9764275229170864e+117 , 4.405849733273784e+120 , 3.9125952982375955e+123 , 3.4880244724478924e+126 ] ,  [ 1.311490084075155e+55 , 6.755424018441362e+58 , 1.9024192668720761e+62 , 3.891044476608491e+65 , 6.480565467527167e+68 , 9.345761664054327e+71 , 1.2121024716167904e+75 , 1.4502705047812485e+78 , 1.6300689961997072e+81 , 1.7443308390839507e+84 , 1.7953814731149056e+87 , 1.791667419204425e+90 , 1.7445825156545914e+93 , 1.6660615600654062e+96 , 1.5670530198795045e+99 , 1.4567408929201034e+102 , 1.342303208392785e+105 , 1.22900181970874e+108 , 1.1204428103470092e+111 , 1.0188982231331428e+114 , 9.256236412958904e+116 , 8.411381156146311e+119 , 7.654537173938494e+122 , 6.98254095597315e+125 , 6.3902752844754676e+128 ] ,  [ 1.009847364737869e+57 , 5.4851270039321196e+60 , 1.6251729711404472e+64 , 3.489904701197398e+67 , 6.090861204991694e+70 , 9.18809042430905e+73 , 1.2444466203387213e+77 , 1.5525425959619495e+80 , 1.816912049646954e+83 , 2.0216517760917726e+86 , 2.1609093157202512e+89 , 2.2367914960009467e+92 , 2.256650277698766e+95 , 2.2305528146666533e+98 , 2.1693189693196556e+101 , 2.0832108163496252e+104 , 1.981194950739898e+107 , 1.870635521868129e+110 , 1.7572729466103535e+113 , 1.6453681935249637e+116 , 1.5379252965303622e+119 , 1.4369350069865508e+122 , 1.3436062365887353e+125 , 1.2585686724199207e+128 , 1.1820407070531714e+131 ] ,  [ 7.977794181429165e+58 , 4.563222055512057e+62 , 1.4207144575866468e+66 , 3.1994892673689143e+69 , 5.845317502181497e+72 , 9.214615143814224e+75 , 1.3021550934077852e+79 , 1.6924776301472907e+82 , 2.0606618183565477e+85 , 2.382379862507822e+88 , 2.6426885822930837e+91 , 2.8355945941090447e+94 , 2.9622740004891315e+97 , 3.028832643479214e+100 , 3.0441891344084537e+103 , 3.0183722862151e+106 , 2.9613150572581347e+109 , 2.882105301554531e+112 , 2.78859981369209e+115 , 2.687297451783128e+118 , 2.583378885835059e+121 , 2.4808408563511706e+124 , 2.3826738091250836e+127 , 2.291049685432587e+130 , 2.207500335879748e+133 ] ,  [ 6.462013286957624e+60 , 3.887387438240572e+64 , 1.2702925096197082e+68 , 2.996855334446967e+71 , 5.725577995498899e+74 , 9.423375266125384e+77 , 1.3881950367422648e+81 , 1.8782496428719918e+84 , 2.377405830055009e+87 , 2.853864081378768e+90 , 3.2831143806224226e+93 , 3.6494192404040083e+96 , 3.9454226083096713e+99 , 4.1706750886185303e+102 , 4.329747621595924e+105 , 4.4303895867830075e+108 , 4.4819784924148816e+111 , 4.494351624671336e+114 , 4.4770121488981634e+117 , 4.438652514362242e+120 , 4.3869212345240293e+123 , 4.328361427041995e+126 , 4.268460870456742e+129 , 4.211767453704991e+132 , 4.1620373253966105e+135 ] ,  [ 5.363471028174828e+62 , 3.3892790477291954e+66 , 1.1611170940879101e+70 , 2.866655459694581e+73 , 5.721864874495047e+76 , 9.823244892383569e+79 , 1.5072884530780794e+83 , 2.1213135243830365e+86 , 2.789369302458557e+89 , 3.474293092445526e+92 , 4.1424743962940073e+95 , 4.767345371474821e+98 , 5.3307666407097664e+101 , 5.822825736678314e+104 , 6.240669823275446e+107 , 6.586911534345152e+110 , 6.86799169654635e+113 , 7.092726831714951e+116 , 7.2711457541572e+119 , 7.413636281087675e+122 , 7.530375226129748e+125 , 7.630993108185857e+128 , 7.724420065844387e+131 , 7.8188639113832e+134 , 7.921879965030082e+137 ] ,  [ 4.558950373948604e+64 , 3.02268432529244e+68 , 1.0844721073997962e+72 , 2.7991369549869103e+75 , 5.831693607862728e+78 , 1.0434463097833564e+82 , 1.666333319560654e+85 , 2.4375400286231998e+88 , 3.3273649900509814e+91 , 4.297407473576291e+94 , 5.307291111184073e+97 , 6.31999564723226e+100 , 7.305258263617033e+103 , 8.241128606633043e+106 , 9.114087735094809e+109 , 9.918232961558827e+112 , 1.0653981357280567e+116 , 1.1326634436230656e+119 , 1.1945027566599618e+122 , 1.25203868770111e+125 , 1.3065442995812543e+128 , 1.3593804036043643e+131 , 1.4119564605067609e+134 , 1.465711710834835e+137 , 1.5221130787989554e+140 ] ,  [ 3.966286825335284e+66 , 2.7561124741912133e+70 , 1.0345086847922348e+74 , 2.7889027544029004e+77 , 6.059386941191012e+80 , 1.1290278798055138e+84 , 1.8750507814096243e+87 , 2.8488658783217254e+90 , 4.0343674314285868e+93 , 5.399483667286943e+96 , 6.90292119699297e+99 , 8.500806023681612e+102 , 1.0152069561017567e+106 , 1.1822117343240376e+109 , 1.3484818203864722e+112 , 1.5123129033776663e+115 , 1.672876753081873e+118 , 1.8301327794283307e+121 , 1.984715942589985e+124 , 2.13782407287744e+127 , 2.2911194003047635e+130 , 2.4466526095211616e+133 , 2.606813237824126e+136 , 2.774307513130412e+139 , 2.952163420121992e+142 ] ,  [ 3.529995274548405e+68 , 2.5680994327817615e+72 , 1.0074757969683406e+76 , 2.8342079279452984e+79 , 6.416283151037918e+82 , 1.2439888177676457e+86 , 2.1469413633505308e+89 , 3.385690647647839e+92 , 4.97076220376314e+95 , 6.889771969462091e+98 , 9.112754812324646e+101 , 1.1599098358650013e+105 , 1.430446047381406e+108 , 1.718663496204823e+111 , 2.0209903890728613e+114 , 2.3347845805979182e+117 , 2.6584828276573133e+120 , 2.99165273732036e+123 , 3.334983169271572e+126 , 3.690244830494122e+129 , 4.060246580322626e+132 , 4.448806677165244e+135 , 4.8607529902045894e+138 , 5.301962542900886e+141 , 5.7794487314495245e+144 ] ,  [ 3.212295699839048e+70 , 2.444209046393343e+74 , 1.0012412339452999e+78 , 2.936663658022446e+81 , 6.9216360886382745e+84 , 1.3953018788960755e+88 , 2.500678125287337e+91 , 4.090371573435206e+94 , 6.222114561388664e+97 , 8.926232973214176e+100 , 1.2207737219968259e+104 , 1.605197746117822e+107 , 2.0432074995222534e+110 , 2.5316632147818727e+113 , 3.0676719499873753e+116 , 3.649163837637703e+119 , 4.275335264914053e+122 , 4.946974687178988e+125 , 5.6667011255575045e+128 , 6.439150992968075e+131 , 7.271149138216904e+134 , 8.171897742345105e+137 , 9.153214010997465e+140 , 1.0229845838064357e+144 , 1.141989460878106e+147 ] ,  [ 2.987435000850315e+72 , 2.3751200605457725e+76 , 1.0150150240260275e+80 , 3.10128779933535e+83 , 7.604300687676185e+86 , 1.5926743904289765e+90 , 2.9621311421866697e+93 , 5.022350803598065e+96 , 7.910762241050947e+99 , 1.173947372301309e+103 , 1.6592178015108966e+106 , 2.252652236146089e+109 , 2.9580478226781723e+112 , 3.7781059245522374e+115 , 4.71539693933027e+118 , 5.773319146767147e+121 , 6.956998304208433e+124 , 8.274114044168607e+127 , 9.73566871570674e+130 , 1.1356734512776609e+134 , 1.3157227789729799e+137 , 1.5162768218935856e+140 , 1.7405687652602442e+143 , 1.9926261617827637e+146 , 2.2774247369386884e+149 ] ,  [ 2.8380632508077995e+74 , 2.355430679192773e+78 , 1.0492248671800216e+82 , 3.336893217500516e+85 , 8.505404875923754e+88 , 1.8495433596467862e+92 , 3.5673180206289105e+95 , 6.265733927644511e+98 , 1.0213264949989444e+102 , 1.566948592290515e+105 , 2.2875490005983507e+108 , 3.205116874228355e+111 , 4.339889512878609e+114 , 5.711237627296913e+117 , 7.338923452013988e+120 , 9.244606154293761e+123 , 1.145350870919219e+127 , 1.3996207348382653e+130 , 1.6910550478503753e+133 , 2.0243755125034274e+136 , 2.4054765506437426e+139 , 2.841699245543239e+142 , 3.3421587454654215e+145 , 3.9181445093959427e+148 , 4.58361773675614e+151 ] ]

def uniform_topological_prior_function(tree):
    up=uniform_prior(get_number_of_leaves(tree))
    A=get_number_of_admixes(tree)
    val=1.0/(up.dic_of_counts[up.leaves])[A]
    return log(val)

def calculate_branch_prior(branches, n):
    rate=float(2*n-2)/len(branches)
    return -sum(branches)*rate+log(rate)*len(branches)

def prior(x, num_admixes, p=0.5):
    tree, add=x
    no_leaves=get_number_of_leaves(tree)
    admixtures=get_all_admixture_proportions(tree)
    if not all(prop>=0 and prop <=1 for prop in admixtures):
        return -float('inf')
    branches=get_all_branch_lengths(tree)
    if not all(branch>=0 for branch in branches):
        return -float('inf')
    branch_prior=calculate_branch_prior(branches, no_leaves)
    no_admix_prior=no_admixes(num_admixes, p, len(admixtures),)
    top_prior=uniform_topological_prior_function(tree)
    logsum=branch_prior+no_admix_prior+top_prior-add
    return logsum

def no_admixes(num_admixes, p, admixes, hard_cutoff=20):
    if admixes > hard_cutoff:
        return -float('inf')
    if num_admixes > -1 and num_admixes < admixes:
        return -float('inf')
    return geom.logpmf(admixes+1, 1.0-p)-geom.logcdf(hard_cutoff+1, 1.0-p)

def likelihood(x, emp_cov, b, M=12,nodes=None):
    tree, add= x
    r=emp_cov.shape[0]
    if nodes is None:
        nodes=["s"+str(i) for i in range(1,r+1)]
    par_cov=make_covariance(tree, nodes)
    if par_cov is None:
        print('illegal tree')
        return -float('inf')
    if b is not None:
        par_cov+=b
    if par_cov is None:
        print('illegal tree')
        return -float('inf')
    try:
        d=wishart.logpdf(emp_cov, df=M, scale= (par_cov+add)/M)
    except (ValueError, LinAlgError) as e:
        return -float("inf")
    return d

class posterior_class(object):

    def __init__(self,  emp_cov,  M=10,  multiplier=None,  nodes=None, varcovname = "", num_admixes = -1):
        '''
        M can either be a float - the degrees of freedom in the wishart distribution or the constant variance in the normal approximation of the covariance matrix.
        or M can be a matrix - the same size of emp_cov where each entry is the variance of that entry.
        '''
        self.num_admixes = num_admixes
        self.emp_cov=emp_cov
        self.M=M
        self.p=0.5
        self.nodes=nodes
        self.b=loadtxt(varcovname) * multiplier

    def __call__(self, x):
        prior_value = prior(x,self.num_admixes, p=self.p)
        if prior_value==-float('inf'):
            return -float('inf'), prior_value

        likelihood_value=likelihood(x, self.emp_cov,self.b, self.M, nodes=self.nodes)
        return likelihood_value, prior_value


def tree_dict_to_nx(tree_dict):

    G = nx.DiGraph()

    for node, data in tree_dict.items():
        p0, p1 = data[0], data[1]
        if p0 is not None and p0 not in tree_dict:
            G.add_node(p0)
        if p1 is not None and p1 not in tree_dict:
            G.add_node(p1)

    for node in tree_dict:
        G.add_node(node)

    for node, data in tree_dict.items():
        parent1, parent2, alpha, bl1, bl2 = data[0], data[1], data[2], data[3], data[4]

        if parent1 is not None:
            G.add_edge(parent1, node, branch_length=float(bl1))

        if parent2 is not None:
            G.add_edge(parent2, node, branch_length=float(bl2))
            G.nodes[node]["admixture_proportion"] = float(alpha if alpha is not None else 0.5)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Converted graph is not a DAG.  Check parent pointers.")
    return G


def draw_admixture_graph(G):
    pos = nx.spring_layout(G, seed=48)
    plt.figure(figsize=(10, 6))

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, arrows=True, font_size=10)

    edge_labels = {}
    for u, v, data in G.edges(data=True):
        bl = data.get('branch_length', data.get('weight', ''))
        edge_labels[(u, v)] = f"{bl:.2f}" if isinstance(bl, (float, int)) else bl
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    for node in G.nodes:
        if 'admixture_proportion' in G.nodes[node]:
            prop = G.nodes[node]['admixture_proportion']
            x, y = pos[node]
            plt.text(x, y - 0.1, f"α={prop:.2f}", fontsize=10, ha='center', color='purple')

    plt.title("Admixture Graph (NetworkX)")
    plt.tight_layout()
    plt.show()


def _save_variance_correction(mat: np.ndarray, folder: str) -> str:
    path = os.path.join(folder, "variance_correction.txt")
    np.savetxt(path, mat)
    return path


def compute_posterior_for_dag(allele_counts_path: str,
                              dag: nx.DiGraph,
                              bootstrap_blocksize: int = 1000,
                              cores: int = 1) -> tuple[float, float]:

    with open(allele_counts_path) as fh:
        header = fh.readline().strip().split()
        if "out" not in header:
            raise ValueError("Header must contain outgroup 'out'.")
        out_idx = header.index("out")

        m_rows, t_rows = [], []
        for ln in fh:
            if not ln.strip():
                continue
            minor, total = [], []
            for pair in ln.split():
                a, b = map(float, pair.split(","))
                minor.append(a); total.append(a + b)
            m_rows.append(minor); t_rows.append(total)

    minors  = np.asarray(m_rows, dtype=float).T   # (pop, snp)
    totals  = np.asarray(t_rows, dtype=float).T
    n_pops, n_snps = minors.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.nan_to_num(minors / totals, nan=0.0)

    tree = {n: [None, None, None, None, None, None, None] for n in dag.nodes}
    for u, v, d in dag.edges(data=True):
        bl = d.get("branch_length", d.get("weight"))
        if bl is None:
            raise KeyError(f"Edge {u}->{v} lacks 'branch_length'/'weight'.")
        if tree[v][0] is None:
            tree[v][0], tree[v][3] = u, float(bl)
        elif tree[v][1] is None:
            tree[v][1], tree[v][4] = u, float(bl)
        else:
            raise ValueError(f"Node '{v}' has >2 parents.")

    for node, attrs in dag.nodes(data=True):
        if tree[node][0] and tree[node][1]:
            prop = attrs.get("admix_prop") or attrs.get("admixture_proportion")
            if prop is None:
                raise ValueError(f"Admixture node '{node}' needs 'admix_prop'.")
            tree[node][2] = float(prop)

    for child in tree:
        for pslot, cslot in ((0, 5), (1, 6)):
            par = tree[child][pslot]
            if par and tree[par][cslot] is None:
                tree[par][cslot] = child

    roots = [n for n, at in tree.items() if at[0] is None and at[1] is None]
    if len(roots) != 1:
        raise ValueError("DAG must have exactly one root.")
    root = roots[0]
    for at in tree.values():
        if at[0] == root: at[0] = "r"
        if at[1] == root: at[1] = "r"
    del tree[root]                                 # root itself isn’t stored

    p_out = p[out_idx]
    diff  = p[np.arange(n_pops) != out_idx] - p_out
    emp_cov = diff @ diff.T / diff.shape[1]

    s = np.nanmean(p, axis=0)
    scaler = np.nanmean(s * (1 - s))
    if scaler <= 0:
        raise RuntimeError("Non‑positive heterozygosity scaling factor.")
    emp_cov /= scaler

    n = emp_cov.shape[0]
    multiplier = (n * (n + 1) / 2 - 1) / np.trace(emp_cov)
    emp_cov *= multiplier

    variances = []
    for i in range(n_pops):
        ni, pi = totals[i], p[i]
        good = ni > 1
        vi = np.mean(pi[good] * (1 - pi[good]) / (ni[good] - 1)) if good.any() else 0.0
        variances.append(vi)
    out_var = variances.pop(out_idx)
    B = np.diag(variances) + out_var * np.ones((n, n))
    B /= scaler

    # print('alala')

    tmpdir = tempfile.mkdtemp(prefix="adbayes_")
    try:
        tmp_allele = os.path.join(tmpdir, "temp_input.txt")
        shutil.copyfile(allele_counts_path, tmp_allele)
        varcov_path = _save_variance_correction(B, tmpdir)

        estimator_args = dict(
            reducer="out",
            nodes=header,
            add_variance_correction_to_graph=False,
            save_variance_correction=False
        )

        # print("google")

        df = estimate_degrees_of_freedom_scaled_fast(
            tmp_allele,
            varcovfilename=varcov_path,
            bootstrap_blocksize=bootstrap_blocksize,
            cores=max(1, cores),
            verbose_level="silent",
            est=estimator_args
        )

        # print(df) # have fluctuations

        pops_wo_out = [p for p in header if p != "out"]
        # print(emp_cov, multiplier, df, pops_wo_out, varcov_path)
        # ! cat /tmp/adbayes_7mvqbg1j/variance_correction.txt
        posterior = posterior_class(
            emp_cov=emp_cov,
            M=df,
            multiplier=multiplier,
            nodes=pops_wo_out,
            varcovname=varcov_path,
            num_admixes=-1
        )


        if 'add' in dag.graph:
            add_val = float(dag.graph['add'])
        else:
            add_val = float(expon.rvs())

        # print(add_val)

        # print('olololol')

        result = posterior((tree, add_val))

        # print(tree)

    finally:
        # pass
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result
