# Copyright (C) 2022 Daniel Grose

# This file is part of seqstats.
#
# seqstats is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# seqstats is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with seqstats. If not, see <https://www.gnu.org/licenses/>. 


from sortedcontainers import SortedList
from math import floor,ceil

class mad :
    '''mad is a class for efficiently maintaining the median absolute 
    deviation of a collection of data which is subject to dynamic updates.
    The data is updated using the add and remove member functions both of
    which employ an efficient algorithm for updating the median absolute 
    deviation which has, for n data elements, a computational complexity of
    O(log(n)*log(n)).
    
    For details of the algorithm employed see 
    https://github.com/grosed/seqstats/tree/main/pdf/pseudo-code-library.pdf
    
    Attributes
    ----------
    value : float
        If the seq_mad object has data elements then mad is the
        value of the median absolute deviation, otherwise its 
        value is equal to None
    
    T : SortedList
        A SortedList containing the data stored by the mad object. For 
        information regarding the SortedList data structure see 
        https://grantjenks.com/docs/sortedcontainers/sortedlist.html
        
    Methods
    -------
    add(x)
        Adds the data element x
        
    remove(x)
        removes the data element x  
    '''
    value : float
    T : SortedList
    def __mad_algorithm(self,T) :
        if len(T) == 1 :
            return 0
        mu = (T[ceil(len(T)/2) - 1] + T[floor(len(T)/2)])/2
        a = 0
        b = floor(len(T)/2) - 1
        c = ceil(len(T)/2)
        d = len(T) - 1  
   
        def bisect(T,Ix,Iy,mu) :
            a,b = Ix
            c,d = Iy
            if b - a < 2 :
                if len(T) % 2 == 0 :
                    return (max(mu-T[b],T[c]-mu) + min(mu-T[a],T[d]-mu))/2
                else :
                    return min(max(mu-T[b],T[c]-mu),min(mu-T[a],T[d]-mu))
            (a,b) = (a + floor((b-a)/2), a + ceil((b-a)/2)) 
            (c,d) = (c + floor((d-c)/2), c + ceil((d-c)/2))
            if 4*mu > T[a] + T[b] + T[c] + T[d] :
                Ix = (a,Ix[1])
                Iy = (c,Iy[1])
            else :
                Ix = (Ix[0],b)
                Iy = (Iy[0],d)
            return bisect(T,Ix,Iy,mu)
    
        return bisect(T,(a,b),(c,d),mu)
    
    def __init__(self) :
        '''Creats an empty mad object.'''
        self.T = SortedList()
        self.value = None
        
    def add(self, x : float) -> None : 
        '''Adds a data element x to the mad object and updates the median 
    absolute deviation. The computational complexity of this method 
    is O(log(n)*log(n)) in the size of the data.
           
    Parameters
    ----------
    x : float
        The data element to be added.
        '''
        self.T.add(x)
        self.value = self.__mad_algorithm(self.T)
        return None
    
    def remove(self, x : float) -> None :
        '''Removes a data element x to the mad object and updates the median 
    absolute deviation. The computational complexity of this method 
    is O(log(n)*log(n)) in the size of the data.
        
    If x is not in the data contained in the mad object then an exception
    is thrown.
        
    Parameters
    ----------
    x : float
        The data element to be removed.
        
    Raises
    ------
    ValueError
        If x is not in the data contained in the mad object.
        '''
        self.T.remove(x)
        if len(self.T) == 0 :
            self.value = None
        else :
            self.value = self.__mad_algorithm(self.T)
        return None
