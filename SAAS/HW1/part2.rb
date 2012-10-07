class WrongNumberOfPlayersError < StandardError ; end
class NoSuchStrategyError < StandardError ; end

def rps_result(m1, m2)
  # YOUR CODE HERE
  #Returns :m1 if m1 wins
  #Returns :m2 if m2 wins
  my_arr = [m1,m2]
  #First of all checking that inputs are correct
  my_arr.each {
   |i| raise NoSuchStrategyError unless (i.downcase =~ /[rps]/ && i.length==1)
  }
  my_temp_str = m1 + m2
  return :m1 unless m1 != m2 #In case tie :m1 always wins
  results = case my_temp_str
     when "RS" then :m1 #Rock Beats Scissors
     when "SP" then :m1 #Scissors Beats Paper
     when "PR" then :m1 #Paper Beats Rock
     else           :m2
  end
  return results
end

def rps_game_winner(game)
  #input in the form of: [ ["Armando", "P"], ["Dave", "S"] ]
  raise WrongNumberOfPlayersError unless game.length == 2
  # YOUR CODE HERE
  result = rps_result(game[0][1],game[1][1])
  if result == :m1 then
    return game[0]
  else
    return game[1]
  end
end

def rps_tournament_winner(tournament)
  # YOUR CODE HERE
  # Checking if we have a game in hand?
  if(tournament[0][0].class() != Array) then
    #we should have a string in hand...
    return rps_game_winner(tournament)
  end
  m1 = rps_tournament_winner(tournament[0])
  m2 = rps_tournament_winner(tournament[1])
  return rps_game_winner([m1,m2])
end
