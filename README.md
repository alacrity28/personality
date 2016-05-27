# Using Data Science to Understand Personality

Welcome to my final project for Galvanize's Data Science Immersive!

This project explores different matrix factorization methods and their effectiveness
in pulling latent personality traits out of a personality data set. Check out my project website here: personalitydatascience.com.

The personality data came from a 140-question personality test and the data set had responses from fifty thousand respondents around the world.  I explored various matrix factorization methods including SVD, PCA, ICA, and clustering (see model.py). To assess the success of each algorithm, I looked at the top survey questions that corresponded to a specific factor. I ultimately choose to use
non-negative matrix factorization (NMF) because it produced the cleanest set of personality traits (both in terms of internal and external category consistency) and because NMF's positive solution allows for more interpretable results.

Upon finding a good algorithm to pull out latent personality traits, I then aimed to reproduce the Big Five personality trait theory (https://en.wikipedia.org/wiki/Big_Five_personality_traits) which is a well-respected personality trait theory because it was created using factor analysis and has been successfully replicated many times.  One main problem with NMF is that it produces a non-unique solution, meaning that it gives a slightly different set of top questions that pertain to a personality trait each time the model is run (though each solution is equally true). I explored research on forcing uniqueness within NMF but there are currently no practical methods that were relevant to scope of this project. Instead, I custom-wrote an "aggregated" NMF which aggregated the results of 1000 NMF models (see aggregatedNMF.py). Using my aggregated NMF model, I was able to replicate 3 out 5 of the Big Five traits in my own data set.

I was also interested in finding the best "natural" personality trait theory. I used NMF and explored pulling different numbers of factors/personality traits out of my data. After qualitative analysis (preserving internal and external category consistency), I choose to pull 8 personality traits out of the data for the ideal personality trait theory for my data. Check out my website (personalitydatascience.com) to check out how my 8 trait personality theory varies by country, gender, and age group in a series of interactive visualizations.
