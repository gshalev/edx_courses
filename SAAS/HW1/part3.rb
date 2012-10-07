def combine_anagrams(words)
  # YOUR CODE HERE
  my_hash = Hash.new {|hash,key| hash[key] = []}
  
  words.each do |w|
    my_key = w.downcase.chars.sort.join
    my_hash[my_key].push(w)   
  end
  my_arr = []
  my_hash.values.each do |val|
    my_arr.push(val)
  end
  return my_arr
end #combine_anagrams
