class Numeric
  @@currencies = {'yen' => 0.013, 'euro' => 1.292, 'rupee' => 0.019, 'dollar' => 1.0 }
  def method_missing(method_id)
    singular_currency = method_id.to_s.gsub( /s$/, '')
    if @@currencies.has_key?(singular_currency)
      self * @@currencies[singular_currency]
    else
      super
    end
  end
  def in(val)
    val = val.to_s.gsub( /s$/, '')
    return self / @@currencies[val] 
  end
end

class String
  def palindrome?
    str = self.downcase.gsub(/[^a-z]/,"")
    return str == str.reverse
  end
end

module Enumerable
  def palindrome?
    if(self.class != Hash) then
      self_arr = self.flat_map{|i| i }
      return self_arr == self_arr.reverse
    end
    return false
  end
end
