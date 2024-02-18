## -- IMPORTS

import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import os;
import pandas;
import sys;

## -- FUNCTIONS

def GetLogicalPath( path ) :

    return path.replace( '\\', '/' );

## -- STATEMENTS

argument_array = sys.argv;
argument_count = len( argument_array ) - 1;

if ( argument_count == 2 ) :

    input_data_file_path = GetLogicalPath( argument_array[ 1 ] );
    output_data_file_path = GetLogicalPath( argument_array[ 2 ] );

    if input_data_file_path.endswith( ".csv" ) :

        try :

            print( "Reading input data :", input_data_file_path );
            data_frame = pandas.read_csv( input_data_file_path, na_filter = False );

            language_code_array = data_frame.columns.tolist();

            for row_index, row in data_frame.iterrows() :

                source_text = "";

                for source_language_code in language_code_array :

                    if row[ source_language_code ] != "" :

                        source_text = row[ source_language_code ];
                        break;

                if source_text != "" :

                    for target_language_code in language_code_array :

                        if row[ target_language_code ] == "" :

                            print( "Loading model..." );
                            model_name = f"Helsinki-NLP/opus-mt-{ source_language_code }-{ target_language_code }";
                            tokenizer = MarianTokenizer.from_pretrained( model_name );
                            model = MarianMTModel.from_pretrained( model_name );

                            print( f"{ source_language_code } : { source_text }" );
                            tokenized_text = tokenizer( [ source_text ], return_tensors = "pt", padding = True );
                            translation = model.generate( **tokenized_text );
                            target_text = tokenizer.decode( translation[ 0 ], skip_special_tokens = True );
                            print( f"{ target_language_code } : { target_text }" );

                            data_frame.at[ row_index, target_language_code ] = target_text;

            print( "Writing output data :", output_data_file_path );
            data_frame.to_csv( output_data_file_path, index = False );

        except Exception as exception :

            print( f"*** { exception }" );

        sys.exit( 0 );

print( f"*** Invalid arguments : { argument_array }" );
print( "Usage: python babel.py input_data.csv output_data.csv" );
