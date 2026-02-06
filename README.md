Multi-label Prototype Selection based on Editing NearestNeighbour Rule

A common method to boost the efficiency of instance-based classifiers, while maintaining  high  accuracy,  is  to  decrease  the  training  set  size  by  substituting  it  with  a smaller,  
representative subset.  This is typically achieved by utilizing data reduction techniques.  The latter enjoy wide applicability in handling single-label classification tasks. This is not the 
case though in cases of multi-label data, where each instance may be associated with not just one, but a number of classes. In the present paper, we adapta popular single-label data reduction 
technique, the Edited Nearest Neighbor (eNN)rule, to handle multi-label data.   In single-label classification,  eNN focuses on the removal of noise and close border instances making the dataset 
clear and the borders well-separated.  The core idea is that instances with a different class than the majority of their neighbors are considered noise and are removed.  In the context of multi-label data, 
label boundaries tend to be ambiguous, and the notion of noise is not clearly defined.  Nevertheless, we hypothesize that instances whose label sets significantly differ from those dominating their local 
neighborhood can be treated as noise and their removal serves to condense the training set.  Based on this principle, we propose three(3) new eNN variations and test them in practice.  
Experimental tests and statistical analysis conducted across nine (9) diverse multi-label datasets indicate that the proposed algorithms reduce significantly the size of the datasets, without compromising on classification accuracy.
