class CartesianProduct
  include Enumerable
  # YOUR CODE HERE
  attr_accessor :arr_i,:arr_j
  def initialize(arr_i,arr_j)
    @arr_i = arr_i
    @arr_j = arr_j
  end
  def each
    return nil if(@arr_i.length == 0 || @arr_j.length == 0)
    my_ret_arr = []
    @arr_i.each do |i|
      @arr_j.each do |j|
        my_ret_arr.push([i,j])
      end
    end  
    my_ret_arr.each do |k| 
      yield k 
    end
  end
end
