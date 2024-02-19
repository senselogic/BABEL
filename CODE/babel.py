## -- IMPORTS

import pandas as pd
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer, MarianMTModel, MarianTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast, T5ForConditionalGeneration, T5Tokenizer;
import os;
import pandas;
import sys;

## -- FUNCTIONS

def GetLogicalPath( path ) :

    return path.replace( '\\', '/' );

##

def GetLanguageName( language_code ) :

    return (
        {
            "ar" : "Arabic",
            "bg" : "Bulgarian",
            "bn" : "Bengali",
            "cs" : "Czech",
            "da" : "Danish",
            "de" : "German",
            "el" : "Greek",
            "en" : "English",
            "es" : "Spanish",
            "et" : "Estonian",
            "fa" : "Persian",
            "fi" : "Finnish",
            "fr" : "French",
            "he" : "Hebrew",
            "hi" : "Hindi",
            "hr" : "Croatian",
            "hu" : "Hungarian",
            "id" : "Indonesian",
            "it" : "Italian",
            "ja" : "Japanese",
            "ko" : "Korean",
            "lt" : "Lithuanian",
            "lv" : "Latvian",
            "ms" : "Malay",
            "nb" : "Norwegian Bokmål",
            "nl" : "Dutch",
            "pl" : "Polish",
            "pt" : "Portuguese",
            "ro" : "Romanian",
            "ru" : "Russian",
            "sk" : "Slovak",
            "sl" : "Slovenian",
            "sr" : "Serbian",
            "sv" : "Swedish",
            "ta" : "Tamil",
            "th" : "Thai",
            "tr" : "Turkish",
            "uk" : "Ukrainian",
            "vi" : "Vietnamese",
            "zh" : "Chinese"
        }
        ).get( language_code, "" );

##

def GetQualifiedLanguageCode( language_code ) :

    return (
        {
            "af" : "af_ZA",
            "ar" : "ar_AR",
            "az" : "az_AZ",
            "bn" : "bn_IN",
            "cs" : "cs_CZ",
            "de" : "de_DE",
            "en" : "en_XX",
            "es" : "es_XX",
            "et" : "et_EE",
            "fa" : "fa_IR",
            "fi" : "fi_FI",
            "fr" : "fr_XX",
            "gl" : "gl_ES",
            "gu" : "gu_IN",
            "he" : "he_IL",
            "hi" : "hi_IN",
            "hr" : "hr_HR",
            "id" : "id_ID",
            "it" : "it_IT",
            "ja" : "ja_XX",
            "ka" : "ka_GE",
            "kk" : "kk_KZ",
            "km" : "km_KH",
            "ko" : "ko_KR",
            "lt" : "lt_LT",
            "lv" : "lv_LV",
            "mk" : "mk_MK",
            "ml" : "ml_IN",
            "mn" : "mn_MN",
            "mr" : "mr_IN",
            "my" : "my_MM",
            "ne" : "ne_NP",
            "nl" : "nl_XX",
            "pl" : "pl_PL",
            "ps" : "ps_AF",
            "pt" : "pt_XX",
            "ro" : "ro_RO",
            "ru" : "ru_RU",
            "si" : "si_LK",
            "sl" : "sl_SI",
            "sv" : "sv_SE",
            "sw" : "sw_KE",
            "ta" : "ta_IN",
            "te" : "te_IN",
            "th" : "th_TH",
            "tl" : "tl_XX",
            "tr" : "tr_TR",
            "uk" : "uk_UA",
            "ur" : "ur_PK",
            "vi" : "vi_VN",
            "xh" : "xh_ZA",
            "zh" : "zh_CN"
        }
        ).get( language_code, "" );

##

def GenerateM2m100Translations( data_frame ) :

    print( "Loading model..." );
    model_name = "facebook/m2m100_1.2B";
    tokenizer = M2M100Tokenizer.from_pretrained( model_name );
    model = M2M100ForConditionalGeneration.from_pretrained( model_name );

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

                    source_language_name = GetLanguageName( source_language_code );
                    target_language_name = GetLanguageName( target_language_code );

                    print( f"{ source_language_name } : { source_text }" );

                    tokenizer.src_lang = source_language_code;
                    inputs = tokenizer( source_text, return_tensors = "pt" );
                    outputs = model.generate( **inputs, forced_bos_token_id = tokenizer.get_lang_id( target_language_code ) );
                    target_text = tokenizer.batch_decode( outputs, skip_special_tokens = True )[ 0 ];

                    print( f"=> { target_language_name } : { target_text }" );

                    data_frame.at[ row_index, target_language_code ] = target_text;

##

def GenerateMbartTranslations( data_frame ) :

    print( "Loading model..." );
    model_name = "facebook/mbart-large-50-many-to-many-mmt";
    model = MBartForConditionalGeneration.from_pretrained( model_name );
    tokenizer = MBart50TokenizerFast.from_pretrained( model_name );

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

                    source_language_name = GetLanguageName( source_language_code );
                    target_language_name = GetLanguageName( target_language_code );

                    qualified_source_language_code = GetQualifiedLanguageCode( source_language_code );
                    qualified_target_language_code = GetQualifiedLanguageCode( target_language_code );

                    print( f"{ source_language_name } : { source_text }" );

                    tokenizer.src_lang = qualified_source_language_code;
                    inputs = tokenizer( source_text, return_tensors = "pt" );
                    outputs = model.generate( **inputs, forced_bos_token_id = tokenizer.lang_code_to_id[ qualified_target_language_code ] );
                    target_text = tokenizer.batch_decode( outputs, skip_special_tokens = True )[ 0 ];

                    print( f"=> { target_language_name } : { target_text }" );

                    data_frame.at[ row_index, target_language_code ] = target_text;

##

def GenerateOpusTranslations( data_frame ) :

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

                    source_language_name = GetLanguageName( source_language_code );
                    target_language_name = GetLanguageName( target_language_code );

                    print( f"{ source_language_name } : { source_text }" );

                    inputs = tokenizer( [ source_text ], return_tensors = "pt", padding = True );
                    outputs = model.generate( **inputs );
                    target_text = tokenizer.decode( outputs[ 0 ], skip_special_tokens = True );

                    print( f"=> { target_language_name } : { target_text }" );

                    data_frame.at[ row_index, target_language_code ] = target_text;

##

def GenerateT5Translations( data_frame ) :

    print( "Loading model..." );
    model_name = "t5-large";
    tokenizer = T5Tokenizer.from_pretrained( model_name, model_max_length = 1024, legacy = False );
    model = T5ForConditionalGeneration.from_pretrained( model_name );

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

                    source_language_name = GetLanguageName( source_language_code );
                    target_language_name = GetLanguageName( target_language_code );
                    task_prefix = f"translate { source_language_name } to { target_language_name }: ";

                    print( f"{ source_language_name } : { source_text }" );

                    input_ids = tokenizer( task_prefix + source_text, return_tensors = "pt" ).input_ids;
                    outputs = model.generate( input_ids, max_new_tokens = 1024 );
                    target_text = tokenizer.decode( outputs[ 0 ], skip_special_tokens = True );

                    print( f"=> { target_language_name } : { target_text }" );

                    data_frame.at[ row_index, target_language_code ] = target_text;

##

def GenerateTranslations( input_data_file_path, output_data_file_path, translation_engine_name ) :

    try :

        print( "Reading input data :", input_data_file_path );
        data_frame = pandas.read_csv( input_data_file_path, na_filter = False );

        if ( translation_engine_name == "m2m100" ) :

            GenerateM2m100Translations( data_frame );

        if ( translation_engine_name == "mbart" ) :

            GenerateMbartTranslations( data_frame );

        elif ( translation_engine_name == "opus" ) :

            GenerateOpusTranslations( data_frame );

        elif ( translation_engine_name == "t5" ) :

            GenerateT5Translations( data_frame );

        print( "Writing output data :", output_data_file_path );
        data_frame.to_csv( output_data_file_path, index = False );

    except Exception as exception :

        print( f"*** { exception }" );


## -- STATEMENTS

argument_array = sys.argv;
argument_count = len( argument_array ) - 1;

if ( argument_count == 3 ) :

    input_data_file_path = GetLogicalPath( argument_array[ 1 ] );
    output_data_file_path = GetLogicalPath( argument_array[ 2 ] );
    translation_engine_name = argument_array[ 3 ];

    if ( input_data_file_path.endswith( ".csv" )
         and output_data_file_path.endswith( ".csv" )
         and ( translation_engine_name == "m2m100"
               or translation_engine_name == "mbart"
               or translation_engine_name == "opus"
               or translation_engine_name == "t5" ) ) :

        GenerateTranslations( input_data_file_path, output_data_file_path, translation_engine_name );

        sys.exit( 0 );

print( f"*** Invalid arguments : { argument_array }" );
print( "Usage: python babel.py input_data.csv output_data.csv m2m100|mbart|opus|t5" );
