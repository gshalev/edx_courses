def palindrome?(str)
  # YOUR CODE HERE
  str = str.downcase.gsub(/[^a-z]/,"")
  return str == str.reverse
end

def count_words(str)
  # YOUR CODE HERE
  my_hash = Hash.new(0) #in order to use the ++ operator
  str.downcase.gsub(/[^a-z\s]/,"")
  .gsub(/\S+/) {  
	|s| my_hash[s]+=1 
  }
  return my_hash
end
