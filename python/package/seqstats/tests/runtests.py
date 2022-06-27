
from seqstats import mad
from sortedcontainers import SortedList
import unittest
from random import random,seed
from statistics import median


class test_mad(unittest.TestCase):
    
    def test_correct_mad_odd_number_of_elements(self) :
        seed(0)
        A = mad()
        X = [random() for i in range(500)]
        for x in X :
            A.add(x)
        X_tilde = median(X)
        self.assertEqual(A.value,median([abs(x-X_tilde) for x in X]))
        
    def test_correct_mad_even_number_of_elements(self) :
        seed(0)
        A = mad()
        X = [random() for i in range(501)]
        for x in X :
            A.add(x)
        X_tilde = median(X)
        self.assertEqual(A.value,median([abs(x-X_tilde) for x in X]))
        
        
    def test_independent_instances(self) :
        seed(0)
        A = mad()
        X = [random() for i in range(500)]
        for x in X :
            A.add(x)
        X_tilde = median(X)
        self.assertEqual(A.value,median([abs(x-X_tilde) for x in X]))
        B = mad()
        seed(1)
        X = [random() for i in range(500)]
        for x in X :
            B.add(x)
        X_tilde = median(X)
        self.assertEqual(B.value,median([abs(x-X_tilde) for x in X]))
        
    def test_add(self) : 
        A = mad()
        A.add(3.14)
        A.add(2.14)
        self.assertEqual(A.T,SortedList([2.14,3.14]))   
        
    
    def test_remove_when_exists(self) : 
        A = mad()
        A.add(3.14)
        A.remove(3.14)
        self.assertEqual(A.T,SortedList([]))
    
    def test_remove_when_not_exists(self) : 
        A = mad()
        A.add(3.14)
        with self.assertRaises(ValueError):
            A.remove(2.14)
            
    def test_remove_when_empty(self) : 
        A = mad()
        with self.assertRaises(ValueError):
            A.remove(3.14)

if __name__ == '__main__':
    unittest.main()
            
