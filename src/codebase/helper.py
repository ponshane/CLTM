# this helper assists us to extract our needed nlp results.
def project_function_for_every_document(tokens, want_stop=False, want_alpha=True, want_lemma=True,
                                       accept_pos = ["NOUN", "PROPN", "ADJ", "ADV", "VERB"],
                                       use_entity=True):
    project_list = [True]*len(tokens)
    project_tokens = []
    # scan every token to check whether match our parameters
    for i, token in enumerate(tokens):
        
        if want_stop != True:
            if token["is_stop"] == True:
                project_list[i] = False
        
        if want_alpha:
            if token["is_alpha"] == False:
                project_list[i] = False
        
        if token["pos_"] not in accept_pos:
            project_list[i] = False
        
        if use_entity:
            if "_" in token["shape_"]:
                project_list[i] = True
        
        if project_list[i] == True:
            if want_lemma == True:
                project_tokens.append(token["lemma_"])
            else:
                project_tokens.append(token["text"])
            
    return project_tokens
